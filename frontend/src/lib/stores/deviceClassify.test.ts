import { describe, it, expect } from 'vitest';
import { classify, kbInset } from './deviceClassify';

describe('classify (size class from viewport arithmetic, spec §4.2)', () => {
	it('phone: width ≤ 600 OR height ≤ 480', () => {
		expect(classify(390, 844).size).toBe('phone'); // portrait phone
		expect(classify(844, 390).size).toBe('phone'); // landscape phone (short)
		expect(classify(1200, 400).size).toBe('phone'); // wide but short
	});
	it('compact: 601–959 wide and tall enough', () => {
		expect(classify(800, 700).size).toBe('compact');
	});
	it('full: ≥ 960 wide and tall enough', () => {
		expect(classify(1440, 900).size).toBe('full');
		expect(classify(960, 600).size).toBe('full');
	});
	it('short is an orthogonal flag (height ≤ 480)', () => {
		expect(classify(844, 390).short).toBe(true);
		expect(classify(1440, 900).short).toBe(false);
	});
});

describe('band boundaries (off-by-one regression guards)', () => {
	it('phone/compact width edge at 600 (≤ vs <)', () => {
		expect(classify(600, 900).size).toBe('phone'); // 600 is still phone
		expect(classify(601, 900).size).toBe('compact'); // 601 crosses into compact
	});
	it('short + phone height edge at 480 (≤ vs <)', () => {
		expect(classify(1000, 480).size).toBe('phone'); // 480 tall is phone by height
		expect(classify(1000, 480).short).toBe(true); // and flagged short
		expect(classify(1000, 481).size).toBe('full'); // 481 tall, 1000 wide → full
		expect(classify(1000, 481).short).toBe(false); // and no longer short
	});
});

describe('kbInset (soft-keyboard overlap)', () => {
	it('is the positive gap between layout height and the visual viewport', () => {
		expect(kbInset(500, 844)).toBe(344); // keyboard covers 344px
		expect(kbInset(844, 844)).toBe(0); // no keyboard
		expect(kbInset(900, 844)).toBe(0); // never negative
	});
});
