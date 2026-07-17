import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import { encodeSyncMsg, decodeSyncMsg, syncHello, onSync, SYNC_TAG_SV } from './syncProtocol';

describe('sync message framing', () => {
	it('round-trips both message kinds', () => {
		for (const kind of ['sv', 'update'] as const) {
			const payload = new Uint8Array([1, 2, 3]);
			const decoded = decodeSyncMsg(encodeSyncMsg({ kind, payload }));
			expect(decoded?.kind).toBe(kind);
			expect(Array.from(decoded!.payload)).toEqual([1, 2, 3]);
		}
	});

	it('rejects empty input and unknown tags', () => {
		expect(decodeSyncMsg(new Uint8Array([]))).toBeNull();
		expect(decodeSyncMsg(new Uint8Array([7, 0]))).toBeNull();
	});

	it('tags a state-vector frame with SYNC_TAG_SV (matches the Rust crate)', () => {
		const doc = new Y.Doc();
		expect(syncHello(doc)[0]).toBe(SYNC_TAG_SV);
	});
});

describe('onSync pairwise handshake', () => {
	it('converges an empty client onto a populated server, then tracks a live edit', () => {
		// Server has a node; client is empty. The symmetric handshake converges the client.
		const server = new Y.Doc();
		const sNodes = server.getMap('nodes');
		const n = new Y.Map();
		n.set('type', 'Oscillator');
		n.set('name', 'osc');
		sNodes.set('1', n);

		const client = new Y.Doc();

		// Client advertises its SV; the server answers with the diff it lacks.
		const clientHello = decodeSyncMsg(syncHello(client))!;
		const replies = onSync(server, clientHello);
		expect(replies.length).toBe(1);
		// Client applies the server's diff → converges.
		for (const r of replies) onSync(client, decodeSyncMsg(r)!);

		const cn = client.getMap('nodes').get('1') as Y.Map<unknown>;
		expect(cn?.get('type')).toBe('Oscillator');
		expect(cn?.get('name')).toBe('osc');

		// A live server edit, relayed as one Update frame, lands on the client.
		const before = Y.encodeStateVector(client);
		(sNodes.get('1') as Y.Map<unknown>).set('name', 'osc2');
		const delta = Y.encodeStateAsUpdate(server, before);
		onSync(client, { kind: 'update', payload: delta });
		expect((client.getMap('nodes').get('1') as Y.Map<unknown>).get('name')).toBe('osc2');
	});

	it('applies remote updates under the given origin so observers can filter echoes', () => {
		const server = new Y.Doc();
		server.getMap('nodes').set('1', new Y.Map());
		const client = new Y.Doc();
		let seenOrigin: unknown = undefined;
		client.on('afterTransaction', (txn: Y.Transaction) => {
			if (txn.changed.size > 0) seenOrigin = txn.origin;
		});
		const full = Y.encodeStateAsUpdate(server);
		onSync(client, { kind: 'update', payload: full }, 'remote');
		expect(seenOrigin).toBe('remote');
	});
});
