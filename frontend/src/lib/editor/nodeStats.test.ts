import { describe, expect, it } from 'vitest';
import { formatUpdateRate, nodeStatsRows } from './nodeStats';

describe('formatUpdateRate', () => {
	it('returns null when there are no stats yet', () => {
		expect(formatUpdateRate(null)).toBeNull();
		expect(formatUpdateRate(undefined)).toBeNull();
	});

	it('renders just the rate, narrowing the decimal on large values', () => {
		expect(formatUpdateRate({ updates_per_second: 12.4, mean_process_ms: 3, total_ticks: 9 })).toBe('12.4 upd/s');
		expect(formatUpdateRate({ updates_per_second: 1000, mean_process_ms: 3, total_ticks: 9 })).toBe('1000 upd/s');
		expect(formatUpdateRate({ updates_per_second: 0, mean_process_ms: 3, total_ticks: 1 })).toBe('0.0 upd/s');
	});
});

describe('nodeStatsRows', () => {
	it('returns an empty list when there are no stats', () => {
		expect(nodeStatsRows(null)).toEqual([]);
		expect(nodeStatsRows(undefined)).toEqual([]);
	});

	it('returns just update rate + process time (no total ticks)', () => {
		expect(nodeStatsRows({ updates_per_second: 12.4, mean_process_ms: 3.12, total_ticks: 1234 })).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' },
			{ label: 'Process time', value: '3.12 ms' }
		]);
	});

	it('narrows decimals as magnitudes grow', () => {
		expect(nodeStatsRows({ updates_per_second: 1000, mean_process_ms: 142.7, total_ticks: 9 })).toEqual([
			{ label: 'Update rate', value: '1000 upd/s' },
			{ label: 'Process time', value: '143 ms' }
		]);
		expect(nodeStatsRows({ updates_per_second: 0, mean_process_ms: 0.04, total_ticks: 1 })).toEqual([
			{ label: 'Update rate', value: '0.0 upd/s' },
			{ label: 'Process time', value: '0.04 ms' }
		]);
	});
});
