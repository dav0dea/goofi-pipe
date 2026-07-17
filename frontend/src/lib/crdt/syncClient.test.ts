import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import { FakeControl } from '$lib/test/fakeControl';
import { SyncClient, REMOTE_ORIGIN } from './syncClient';
import { decodeSyncMsg, encodeSyncMsg, syncHello } from './syncProtocol';
import { encodeEphemeral } from './ephemeral';
import { nodeView, paramValue, setParamValue } from './graphDoc';

/** A stand-in for the manager's authoritative replica, driven by hand in the test. */
function serverWithNode(): Y.Doc {
	const server = new Y.Doc();
	const n = new Y.Map<unknown>();
	n.set('type', 'Oscillator');
	n.set('name', 'osc');
	(server.getMap('nodes') as Y.Map<unknown>).set('1', n);
	return server;
}

/** Seed the same node into a client's own replica so leaf-writes land. */
function seedClientNode(client: SyncClient): void {
	const n = new Y.Map<unknown>();
	n.set('type', 'Oscillator');
	n.set('name', 'osc');
	(client.doc.getMap('nodes') as Y.Map<unknown>).set('1', n);
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

	it('re-advertises its state vector on every reconnect', () => {
		// Regression for a permanent-desync bug: advertising only once at mount leaves the
		// replica diverged after a WS drop, since the manager re-sends only its own SV on
		// reconnect (which a reader answers with an empty diff). The client must re-advertise.
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();
		expect(ctl.sentSyncFrames.length).toBe(1); // initial advertise (onConnect fires true)

		ctl.setConnected(false); // WS drops
		ctl.setConnected(true); // ...and reconnects
		expect(ctl.sentSyncFrames.length).toBe(2); // re-advertised, so the manager re-syncs us

		const msg = decodeSyncMsg(ctl.sentSyncFrames.at(-1)!);
		expect(msg?.kind).toBe('sv');
	});

	it('publishes ephemeral state framed with its client id', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.publishEphemeral({ cursor: [3, 4] });
		const msg = decodeSyncMsg(ctl.sentSyncFrames.at(-1)!);
		expect(msg?.kind).toBe('ephemeral');
	});

	it('routes an inbound ephemeral frame to the store (never the doc) and self-filters', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		let notified = 0;
		client.setEphemeralListener(() => notified++);

		// A peer's presence arrives → tracked in the store.
		const peer = encodeSyncMsg({ kind: 'ephemeral', payload: encodeEphemeral({ client: 999, state: { name: 'peer' } }) });
		client.onFrame(peer);
		expect(client.ephemeral.get(999)).toEqual({ name: 'peer' });
		expect(notified).toBe(1);
		// The doc is untouched by ephemeral frames.
		expect(client.doc.getMap('nodes').size).toBe(0);

		// Our own echoed frame is ignored (self-filter).
		const own = encodeSyncMsg({ kind: 'ephemeral', payload: encodeEphemeral({ client: client.clientId, state: { x: 1 } }) });
		client.onFrame(own);
		expect(client.ephemeral.get(client.clientId)).toBeUndefined();
	});

	it('commit skips a guarded no-op even after prior overwrites (tombstone-proof)', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		seedClientNode(client);

		// Two real writes: the second overwrites the first, tombstoning the old value. An update
		// payload ALWAYS carries the doc's full delete set, so from here `encodeStateAsUpdate`
		// is never the 2-byte empty update — a byte-length skip check would be defeated.
		client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 1));
		client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 2));
		ctl.sentSyncFrames.length = 0;

		// The guarded writer makes this mutate a no-op (value already 2) → no transaction change →
		// nothing to broadcast, despite the tombstones sitting in the doc.
		const sent = client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 2));
		expect(sent).toBe(false);
		expect(ctl.sentSyncFrames.length).toBe(0);
	});

	it('commit broadcasts a real write as a precise delta (not the whole delete set)', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		seedClientNode(client);
		// Prior overwrite → tombstone present.
		client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 1));
		client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 2));

		// A server replica caught up to the client's current state.
		const server = new Y.Doc();
		Y.applyUpdate(server, Y.encodeStateAsUpdate(client.doc));
		ctl.sentSyncFrames.length = 0;

		const sent = client.commit((doc) => setParamValue(doc, '1', 'common', 'freq', 9));
		expect(sent).toBe(true);
		expect(ctl.sentSyncFrames.length).toBe(1);
		const msg = decodeSyncMsg(ctl.sentSyncFrames[0]);
		expect(msg?.kind).toBe('update');
		// Applying the broadcast delta advances the (already-synced) server to the new value.
		Y.applyUpdate(server, msg!.payload);
		expect(paramValue(server, '1', 'common', 'freq')).toBe(9);
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
