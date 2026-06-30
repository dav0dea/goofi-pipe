import { describe, it, expect } from 'vitest';
import { makeLUT, makeLUTCache } from './colormaps';

describe('makeLUTCache — single-slot LUT memo', () => {
	it('returns the SAME reference for repeated same-name calls', () => {
		const lutFor = makeLUTCache();
		const a = lutFor('gray');
		const b = lutFor('gray');
		expect(a).toBe(b); // memoized — not rebuilt
	});

	it('rebuilds when the requested name changes', () => {
		const lutFor = makeLUTCache();
		const gray = lutFor('gray');
		const warm = lutFor('coolwarm');
		expect(warm).not.toBe(gray);
		// the rebuilt LUT matches a fresh makeLUT for that name
		expect(Array.from(warm)).toEqual(Array.from(makeLUT('coolwarm')));
		// switching back rebuilds again (single slot, not a multi-entry cache)
		const grayAgain = lutFor('gray');
		expect(grayAgain).not.toBe(gray);
		expect(Array.from(grayAgain)).toEqual(Array.from(makeLUT('gray')));
	});
});
