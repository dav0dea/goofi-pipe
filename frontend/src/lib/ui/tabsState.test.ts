import { describe, it, expect } from 'vitest';
import { resolveActive, nextIndex, type TabItem, type ArrowKey } from './tabsState';

// The Tabs primitive's two pure decisions (spec §2.4, §3): which tab is active given the `active`
// prop (active-resolution) and where an arrow key moves it (next-index-on-arrow). Kept pure +
// unit-tested so the connected-bar component stays thin glue and the selection rule has one SSOT.
const items: TabItem[] = [
	{ id: 'a', label: 'A' },
	{ id: 'b', label: 'B' },
	{ id: 'c', label: 'C' }
];

describe('resolveActive', () => {
	it('returns the given active id when it is one of the items', () => {
		expect(resolveActive(items, 'b')).toBe('b');
	});

	it('falls back to the first item when active is unset', () => {
		expect(resolveActive(items, undefined)).toBe('a');
	});

	it('falls back to the first item when active names no item (stale id)', () => {
		expect(resolveActive(items, 'gone')).toBe('a');
	});

	it('returns undefined when there are no items', () => {
		expect(resolveActive([], 'a')).toBeUndefined();
		expect(resolveActive([], undefined)).toBeUndefined();
	});
});

describe('nextIndex', () => {
	const count = 3;

	it('ArrowRight advances one, wrapping past the end', () => {
		expect(nextIndex(0, count, 'ArrowRight')).toBe(1);
		expect(nextIndex(1, count, 'ArrowRight')).toBe(2);
		expect(nextIndex(2, count, 'ArrowRight')).toBe(0); // wrap
	});

	it('ArrowLeft retreats one, wrapping past the start', () => {
		expect(nextIndex(2, count, 'ArrowLeft')).toBe(1);
		expect(nextIndex(1, count, 'ArrowLeft')).toBe(0);
		expect(nextIndex(0, count, 'ArrowLeft')).toBe(2); // wrap
	});

	it('Home jumps to the first, End to the last', () => {
		expect(nextIndex(2, count, 'Home')).toBe(0);
		expect(nextIndex(0, count, 'End')).toBe(count - 1);
	});

	it('with no current selection, ArrowRight/Home start at the first, ArrowLeft/End at the last', () => {
		expect(nextIndex(-1, count, 'ArrowRight')).toBe(0);
		expect(nextIndex(-1, count, 'Home')).toBe(0);
		expect(nextIndex(-1, count, 'ArrowLeft')).toBe(count - 1);
		expect(nextIndex(-1, count, 'End')).toBe(count - 1);
	});

	it('returns -1 when there are no tabs to move among', () => {
		const keys: ArrowKey[] = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
		for (const k of keys) expect(nextIndex(0, 0, k)).toBe(-1);
	});
});
