import { describe, it, expect } from 'vitest';
import { TAGS } from './vocab';

describe('the tag vocabulary', () => {
	it('is the backend list, and holds the role facet', () => {
		for (const t of ['input', 'output', 'generator', 'transform', 'analysis', 'control']) {
			expect(TAGS).toContain(t);
		}
	});
});
