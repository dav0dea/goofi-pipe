import { describe, it, expect } from 'vitest';
import { readEnvelope, envelopeBand } from './envelope';

describe('readEnvelope', () => {
	it('reads a last-axis envelope descriptor', () => {
		const meta = { reduced: { '0': { orig_len: 256, method: 'envelope' } } };
		expect(readEnvelope(meta, 1)).toEqual({ origLen: 256 });
	});

	it('reads the LAST dim for a 2-D frame', () => {
		const meta = { reduced: { '1': { orig_len: 500, method: 'envelope' } } };
		expect(readEnvelope(meta, 2)).toEqual({ origLen: 500 });
	});

	it('returns null when the last axis is not envelope-reduced', () => {
		expect(readEnvelope({ reduced: { '0': { orig_len: 9, method: 'subsample' } } }, 1)).toBeNull();
		expect(readEnvelope({ reduced: {} }, 1)).toBeNull();
		expect(readEnvelope({}, 1)).toBeNull();
		expect(readEnvelope(null, 1)).toBeNull();
		expect(readEnvelope({ reduced: { '0': { orig_len: 256, method: 'envelope' } } }, 0)).toBeNull();
	});
});

describe('envelopeBand', () => {
	it('passes min/max pairs through and reconstructs the original x-span', () => {
		// One channel, W=2 bins over 8 original samples: [min0,max0,min1,max1].
		const band = envelopeBand([[1, 4, 5, 8]], 8);
		expect(band.ys).toEqual([[1, 4, 5, 8]]);
		// Bin 0 spans samples [0,4) → x slots 0 and 3; bin 1 spans [4,8) → 4 and 7.
		expect(band.xs).toEqual([0, 3, 4, 7]);
	});

	it('applies the log-x base offset', () => {
		const band = envelopeBand([[1, 4, 5, 8]], 8, 1);
		expect(band.xs).toEqual([1, 4, 5, 8]);
	});

	it('handles multiple channels sharing one x grid', () => {
		const band = envelopeBand(
			[
				[0, 1, 2, 3],
				[10, 11, 12, 13]
			],
			100
		);
		expect(band.ys.length).toBe(2);
		expect(band.xs.length).toBe(4);
		expect(band.ys[1]).toEqual([10, 11, 12, 13]);
	});
});
