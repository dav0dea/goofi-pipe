import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import { FakeControl } from '$lib/test/fakeControl';
import { SyncClient, REMOTE_ORIGIN } from './syncClient';
import { decodeSyncMsg, encodeSyncMsg, syncHello } from './syncProtocol';
import { nodeView } from './graphDoc';

/** A stand-in for the manager's authoritative replica, driven by hand in the test. */
function serverWithNode(): Y.Doc {
	const server = new Y.Doc();
	const n = new Y.Map<unknown>();
	n.set('type', 'Oscillator');
	n.set('name', 'osc');
	(server.getMap('nodes') as Y.Map<unknown>).set('1', n);
	return server;
}

describe('SyncClient', () => {
	it('advertises its state vector on start', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();
		expect(ctl.sentSyncFrames.length).toBe(1);
		const msg = decodeSyncMsg(ctl.sentSyncFrames[0]);
		expect(msg?.kind).toBe('sv');
	});

	it('converges the doc when the manager replies to its state vector', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();

		// The manager receives the client's SV (the frame it just sent) and answers with a diff.
		const server = serverWithNode();
		const clientSv = decodeSyncMsg(ctl.sentSyncFrames[0])!;
		const reply = encodeSyncMsg({
			kind: 'update',
			payload: Y.encodeStateAsUpdate(server, clientSv.payload)
		});

		ctl.emitSync(reply); // manager → client
		expect(nodeView(client.doc, '1')).toMatchObject({ type: 'Oscillator', name: 'osc' });
	});

	it('replies to a manager state-vector with the update it lacks', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		// Give the client a node the (empty) manager does not have.
		const n = new Y.Map<unknown>();
		n.set('type', 'Buffer');
		(client.doc.getMap('nodes') as Y.Map<unknown>).set('9', n);
		client.start();
		ctl.sentSyncFrames.length = 0; // ignore the hello

		// Manager advertises its (empty) SV; client must answer with an Update carrying node 9.
		const server = new Y.Doc();
		ctl.emitSync(syncHello(server));
		const out = decodeSyncMsg(ctl.sentSyncFrames.at(-1)!);
		expect(out?.kind).toBe('update');
		Y.applyUpdate(server, out!.payload);
		expect(nodeView(server, '9')).toMatchObject({ type: 'Buffer' });
	});

	it('stamps applied remote updates with REMOTE_ORIGIN', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		let origin: unknown;
		client.doc.on('afterTransaction', (txn: Y.Transaction) => {
			if (txn.changed.size > 0) origin = txn.origin;
		});
		client.start();
		const server = serverWithNode();
		ctl.emitSync(encodeSyncMsg({ kind: 'update', payload: Y.encodeStateAsUpdate(server) }));
		expect(origin).toBe(REMOTE_ORIGIN);
	});
});
