import { describe, it, expect } from 'vitest';
import { dtypeColor } from './categoryColor';

describe('dtypeColor is untouched (load-bearing --dtype channel)', () => {
	it('still maps dtypes to their tokens', () => {
		expect(dtypeColor('ARRAY')).toBe('var(--dtype-array)');
		expect(dtypeColor('STRING')).toBe('var(--dtype-string)');
		expect(dtypeColor('TABLE')).toBe('var(--dtype-table)');
	});
});
