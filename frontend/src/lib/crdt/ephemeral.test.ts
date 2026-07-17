import { describe, it, expect } from 'vitest';
import { encodeEphemeral, decodeEphemeral, EphemeralStore } from './ephemeral';

describe('ephemeral framing', () => {
	it('round-trips a state update', () => {
		const enc = encodeEphemeral({ client: 7, state: { cursor: [1, 2], label: 'a' } });
		const dec = decodeEphemeral(enc);
		expect(dec).toEqual({ client: 7, state: { cursor: [1, 2], label: 'a' } });
	});

	it('encodes a removal as an empty state', () => {
		const dec = decodeEphemeral(encodeEphemeral({ client: 9, state: null }));
		expect(dec).toEqual({ client: 9, state: null });
	});

	it('rejects a too-short or malformed payload', () => {
		expect(decodeEphemeral(new Uint8Array([1, 2, 3]))).toBeNull();
		// 8-byte client id + invalid JSON body
		const bad = new Uint8Array(12);
		bad.set([123, 34], 8); // `{"` — truncated JSON
		expect(decodeEphemeral(bad)).toBeNull();
	});

	it('uses little-endian u64 client ids (matches the Rust relay framing)', () => {
		const enc = encodeEphemeral({ client: 0x07, state: 1 });
		expect(enc[0]).toBe(0x07); // low byte first
		expect(enc[7]).toBe(0x00);
	});
});

describe('EphemeralStore', () => {
	it('tracks remote clients and self-filters the local id', () => {
		const store = new EphemeralStore(1);
		store.apply({ client: 2, state: { name: 'b' } });
		store.apply({ client: 1, state: { name: 'me' } }); // own id — ignored
		expect(store.size).toBe(1);
		expect(store.get(2)).toEqual({ name: 'b' });
		expect(store.get(1)).toBeUndefined();
		expect(store.entries()).toEqual([[2, { name: 'b' }]]);
	});

	it('latest-wins per client and removes on null', () => {
		const store = new EphemeralStore(1);
		store.apply({ client: 3, state: { drag: 10 } });
		store.apply({ client: 3, state: { drag: 20 } });
		expect(store.get(3)).toEqual({ drag: 20 });
		store.apply({ client: 3, state: null }); // client 3 left
		expect(store.get(3)).toBeUndefined();
		expect(store.size).toBe(0);
	});
});
