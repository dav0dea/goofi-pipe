import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(__dirname, '../../..');
const FONTS = path.join(ROOT, 'static/fonts');
const appCss = (): string => fs.readFileSync(path.join(ROOT, 'src/app.css'), 'utf8');

describe('the shipped faces (two-face typography, D-T1)', () => {
	// The stylesheet named JetBrains Mono for a month while every machine rendered an unknown
	// fontconfig fallback — the fonts must BE here, valid, and declared, or the whole optical
	// centring story silently reverts to nondeterminism.
	for (const file of ['InterVariable.woff2', 'JetBrainsMono.woff2']) {
		it(`${file} is present and is a real woff2`, () => {
			const p = path.join(FONTS, file);
			expect(fs.existsSync(p), `${file} vendored`).toBe(true);
			const head = fs.readFileSync(p).subarray(0, 4).toString('latin1');
			expect(head, `${file} magic bytes`).toBe('wOF2');
			expect(fs.statSync(p).size).toBeGreaterThan(50_000);
		});
	}
	it('both faces ship their OFL licence', () => {
		expect(fs.existsSync(path.join(FONTS, 'LICENSE-Inter.txt'))).toBe(true);
		expect(fs.existsSync(path.join(FONTS, 'LICENSE-JetBrainsMono.txt'))).toBe(true);
	});
	it('app.css declares both @font-face blocks against the vendored files', () => {
		const css = appCss();
		expect(css).toMatch(/@font-face[^}]*'Inter'[^}]*\/fonts\/InterVariable\.woff2/s);
		expect(css).toMatch(/@font-face[^}]*'JetBrains Mono'[^}]*\/fonts\/JetBrainsMono\.woff2/s);
		expect(css.match(/font-display:\s*block/g)?.length, 'block, not swap — a fallback flash IS the bug').toBe(2);
	});
	it('the two tokens exist and lead with the shipped faces', () => {
		const css = appCss();
		expect(css).toMatch(/--font-sans:\s*\n?\s*'Inter'/);
		expect(css).toMatch(/--font-mono:\s*\n?\s*'JetBrains Mono'/);
	});
});
