import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { contrastRatio, relativeLuminance } from './contrast';

const css = readFileSync(fileURLToPath(new URL('../../app.css', import.meta.url)), 'utf8');

/** Resolve a `--token` to its #rrggbb value from the :root block (one level of var() alias). */
function token(name: string): string {
	const re = new RegExp(`--${name}\\s*:\\s*([^;]+);`);
	const m = re.exec(css);
	if (!m) throw new Error(`token --${name} not found`);
	const v = m[1].trim();
	const alias = /^var\(--([a-z0-9-]+)\)$/i.exec(v);
	return alias ? token(alias[1]) : v;
}

describe('elevation ladder', () => {
	const steps = ['bg', 'surface-1', 'surface-2', 'surface-3', 'surface-4'].map(token);
	it('is monotonically lighter', () => {
		for (let i = 1; i < steps.length; i++) {
			expect(relativeLuminance(steps[i])).toBeGreaterThan(relativeLuminance(steps[i - 1]));
		}
	});
	it('steps each ≥ 1.08:1 (a real visible increment, unlike the old flat ladder)', () => {
		for (let i = 1; i < steps.length; i++) {
			expect(contrastRatio(steps[i], steps[i - 1])).toBeGreaterThanOrEqual(1.08);
		}
	});
});

describe('text tiers on --bg', () => {
	const bg = token('bg');
	it('--text ≥ 7:1', () => expect(contrastRatio(token('text'), bg)).toBeGreaterThanOrEqual(7));
	it('--text-dim ≥ 4.5:1', () => expect(contrastRatio(token('text-dim'), bg)).toBeGreaterThanOrEqual(4.5));
	it('--text-muted ≥ 3:1 on --bg and on --surface-2', () => {
		expect(contrastRatio(token('text-muted'), bg)).toBeGreaterThanOrEqual(3);
		expect(contrastRatio(token('text-muted'), token('surface-2'))).toBeGreaterThanOrEqual(3);
	});
	it('--text-faint is bridged to the muted value (no 3.0:1 failure)', () => {
		expect(relativeLuminance(token('text-faint'))).toBeCloseTo(relativeLuminance(token('text-muted')), 5);
	});
});

describe('borders read as a light hairline, never an inverted groove', () => {
	it('--border and --border-strong are lighter than --surface-4', () => {
		const s4 = relativeLuminance(token('surface-4'));
		expect(relativeLuminance(token('border'))).toBeGreaterThan(s4);
		expect(relativeLuminance(token('border-strong'))).toBeGreaterThan(s4);
	});
});

describe('semantic + dtype colours stay legible on the lightest surface', () => {
	const s4 = token('surface-4');
	for (const t of ['success', 'warning', 'danger', 'dtype-array', 'dtype-string', 'dtype-table']) {
		it(`--${t} ≥ 3:1 on --surface-4`, () => expect(contrastRatio(token(t), s4)).toBeGreaterThanOrEqual(3));
	}
});
