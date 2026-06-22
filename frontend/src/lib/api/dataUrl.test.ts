import { describe, expect, it } from 'vitest';
import { dataUrl } from './dataUrl';

describe('dataUrl', () => {
	it('builds the /data/<node>/<slot>/<kind> WS URL', () => {
		expect(dataUrl('ws:', 'localhost:8000', 'osc0', 'out', 'line')).toBe(
			'ws://localhost:8000/data/osc0/out/line'
		);
	});

	it('url-encodes node, slot, and kind segments', () => {
		expect(dataUrl('wss:', 'h', 'sub0::psd0', 'a/b', 'image')).toBe(
			'wss://h/data/sub0%3A%3Apsd0/a%2Fb/image'
		);
	});
});
