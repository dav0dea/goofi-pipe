import { describe, expect, it } from 'vitest';
import { formatUpdateRate, nodeStatsRows } from './nodeStats';

describe('formatUpdateRate', () => {
	it('returns null when there are no stats yet', () => {
		expect(formatUpdateRate(null)).toBeNull();
		expect(formatUpdateRate(undefined)).toBeNull();
	});

	it('renders just the rate, narrowing the decimal on large values', () => {
		expect(formatUpdateRate({ updates_per_second: 12.4 })).toBe('12.4 upd/s');
		expect(formatUpdateRate({ updates_per_second: 1000 })).toBe('1000 upd/s');
		expect(formatUpdateRate({ updates_per_second: 0 })).toBe('0.0 upd/s');
	});
});

describe('nodeStatsRows', () => {
	it('returns an empty list when there are no stats', () => {
		expect(nodeStatsRows(null)).toEqual([]);
		expect(nodeStatsRows(undefined)).toEqual([]);
	});

	it('returns the update rate — the one metric the engine measures', () => {
		expect(nodeStatsRows({ updates_per_second: 12.4 })).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' }
		]);
	});

	it('narrows decimals as magnitudes grow', () => {
		expect(nodeStatsRows({ updates_per_second: 1000 })).toEqual([{ label: 'Update rate', value: '1000 upd/s' }]);
		expect(nodeStatsRows({ updates_per_second: 0 })).toEqual([{ label: 'Update rate', value: '0.0 upd/s' }]);
	});
});
