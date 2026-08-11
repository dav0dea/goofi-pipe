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

describe('the drop row', () => {
	it('reports the slot drop rate beside the node update rate', () => {
		expect(nodeStatsRows({ updates_per_second: 12.4 }, 3.2)).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' },
			{ label: 'Dropped', value: '3.2/s' }
		]);
	});

	it('shows a healthy stream as 0.0/s rather than hiding the row', () => {
		// A stream that is running and dropping nothing is worth SAYING — it is the reading a user
		// checks against. Hiding the row at zero makes its absence ambiguous with "no stream".
		expect(nodeStatsRows({ updates_per_second: 30 }, 0)).toEqual([
			{ label: 'Update rate', value: '30.0 upd/s' },
			{ label: 'Dropped', value: '0.0/s' }
		]);
	});

	it('omits the drop row entirely when no stream is bound', () => {
		expect(nodeStatsRows({ updates_per_second: 12.4 }, null)).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' }
		]);
		expect(nodeStatsRows({ updates_per_second: 12.4 })).toEqual([
			{ label: 'Update rate', value: '12.4 upd/s' }
		]);
	});
});
