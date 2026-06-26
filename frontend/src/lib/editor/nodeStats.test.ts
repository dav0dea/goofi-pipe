import { describe, expect, it } from 'vitest';
import { formatNodeStats, nodeStatsRows } from './nodeStats';

describe('formatNodeStats', () => {
	it('returns null when there are no stats yet', () => {
		expect(formatNodeStats(null)).toBeNull();
		expect(formatNodeStats(undefined)).toBeNull();
	});

	it('renders a compact "rate · duration" overlay', () => {
		expect(formatNodeStats({ updates_per_second: 12.4, mean_process_ms: 3.12, total_ticks: 40 })).toBe(
			'12.4 upd/s · 3.12 ms'
		);
	});

	it('drops the decimal on large rates and durations to stay compact', () => {
		expect(formatNodeStats({ updates_per_second: 1000.0, mean_process_ms: 142.7, total_ticks: 9 })).toBe(
			'1000 upd/s · 143 ms'
		);
	});

	it('keeps a tenths place on mid-range durations and one decimal sub-ms precision', () => {
		expect(formatNodeStats({ updates_per_second: 5, mean_process_ms: 24.6, total_ticks: 3 })).toBe(
			'5.0 upd/s · 24.6 ms'
		);
		expect(formatNodeStats({ updates_per_second: 0, mean_process_ms: 0.04, total_ticks: 1 })).toBe(
			'0.0 upd/s · 0.04 ms'
		);
	});
});

describe('nodeStatsRows', () => {
	it('returns an empty list when there are no stats', () => {
		expect(nodeStatsRows(null)).toEqual([]);
	});

	it('returns labelled rows for the inspector', () => {
		expect(nodeStatsRows({ updates_per_second: 12.4, mean_process_ms: 3.12, total_ticks: 1234 })).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' },
			{ label: 'Process time', value: '3.12 ms' },
			{ label: 'Total ticks', value: '1234' }
		]);
	});
});
