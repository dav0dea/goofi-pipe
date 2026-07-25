import { describe, it, expect } from 'vitest';
import { SERIES } from './palette';

describe('canvas series palette', () => {
	it('SERIES is a single non-empty list of #rrggbb (no 8-vs-7 drift)', () => {
		expect(SERIES.length).toBeGreaterThan(0);
		for (const c of SERIES) expect(c).toMatch(/^#[0-9a-fA-F]{6}$/);
	});
});
