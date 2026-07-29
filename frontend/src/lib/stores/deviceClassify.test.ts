import { describe, it, expect } from 'vitest';
import { kbInset } from './deviceClassify';

describe('kbInset (soft-keyboard overlap)', () => {
	it('is the positive gap between layout height and the visual viewport', () => {
		expect(kbInset(500, 844)).toBe(344); // keyboard covers 344px
		expect(kbInset(844, 844)).toBe(0); // no keyboard
		expect(kbInset(900, 844)).toBe(0); // never negative
	});
});
