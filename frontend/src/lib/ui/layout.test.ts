import { describe, it, expect } from 'vitest';
import {
	resolveSpace,
	alignItems,
	justifyContent,
	type AlignSetting,
	type JustifySetting
} from './layout';

// The layout primitives (Stack/Row) resolve their `gap`/`align`/`justify` props through this
// one pure module (spec §2.3, §3): a spacing-scale key → an F `--space-N` token (never a literal),
// and the short align/justify unions → their CSS flexbox values. Kept unit-tested so the mapping
// is one source of truth the components apply, not re-derived (and drifting) per component.
describe('resolveSpace', () => {
	it('maps a numeric scale key to the F --space token', () => {
		expect(resolveSpace(1)).toBe('var(--space-1)');
		expect(resolveSpace(4)).toBe('var(--space-4)');
		expect(resolveSpace(8)).toBe('var(--space-8)');
	});

	it('maps a numeric-string scale key to the F --space token', () => {
		expect(resolveSpace('4')).toBe('var(--space-4)');
	});

	it('resolves 0 to a literal zero (no --space-0 token exists)', () => {
		expect(resolveSpace(0)).toBe('0');
		expect(resolveSpace('0')).toBe('0');
	});

	it('passes an already-resolved token or length through unchanged', () => {
		expect(resolveSpace('var(--space-4)')).toBe('var(--space-4)');
		expect(resolveSpace('1rem')).toBe('1rem');
	});

	it('never emits a literal for an in-scale key — always the token', () => {
		for (const n of [1, 2, 3, 4, 5, 6, 7, 8] as const) {
			expect(resolveSpace(n)).toBe(`var(--space-${n})`);
		}
	});

	it('is pure — identical input yields identical output', () => {
		expect(resolveSpace(6)).toBe(resolveSpace(6));
	});
});

describe('alignItems', () => {
	it('maps the short union to CSS align-items values', () => {
		const cases: Record<AlignSetting, string> = {
			stretch: 'stretch',
			start: 'flex-start',
			center: 'center',
			end: 'flex-end',
			baseline: 'baseline'
		};
		for (const [key, css] of Object.entries(cases)) {
			expect(alignItems(key as AlignSetting)).toBe(css);
		}
	});
});

describe('justifyContent', () => {
	it('maps the short union to CSS justify-content values', () => {
		const cases: Record<JustifySetting, string> = {
			start: 'flex-start',
			center: 'center',
			end: 'flex-end',
			between: 'space-between',
			around: 'space-around',
			evenly: 'space-evenly'
		};
		for (const [key, css] of Object.entries(cases)) {
			expect(justifyContent(key as JustifySetting)).toBe(css);
		}
	});
});
