import { describe, it, expect } from 'vitest';
import { categoryColor, dtypeColor } from './categoryColor';

describe('categoryColor is neutralised (node categories are gone, D4)', () => {
	it('returns the neutral --text-muted for every former category', () => {
		for (const c of ['analysis', 'array', 'inputs', 'misc', 'outputs', 'signal', 'viewer', 'subpatch', undefined, null, '']) {
			expect(categoryColor(c)).toBe('var(--text-muted)');
		}
	});
});

describe('dtypeColor is untouched (load-bearing --dtype channel)', () => {
	it('still maps dtypes to their tokens', () => {
		expect(dtypeColor('ARRAY')).toBe('var(--dtype-array)');
		expect(dtypeColor('STRING')).toBe('var(--dtype-string)');
		expect(dtypeColor('TABLE')).toBe('var(--dtype-table)');
	});
});
