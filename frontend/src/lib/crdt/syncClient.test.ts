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



	it('reset() replaces the replica with a fresh empty doc that pushes nothing stale', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();
		// Give the replica session-A content (a node the OLD backend had).
		seedClientNode(client);
		const oldDoc = client.doc;
		expect(client.doc.getMap('nodes').size).toBe(1);

		client.reset();

		// A brand-new, EMPTY doc with a fresh client id — the stale session-A content is gone.
		expect(client.doc).not.toBe(oldDoc);
		expect(client.doc.getMap('nodes').size).toBe(0);

		// Answering a NEW backend's state vector must serialize nothing (no stale push into the new
		// session): the fresh doc holds no content the server lacks.
		const serverB = new Y.Doc(); // the new session's (empty-ish) replica
		ctl.sentSyncFrames.length = 0;
		client.onFrame(syncHello(serverB)); // server advertises its SV → client answers
		const answer = ctl.sentSyncFrames.map(decodeSyncMsg).find((m) => m?.kind === 'update');
		// The update (if any) carries an EMPTY payload — the fresh doc has nothing to push.
		if (answer) expect(answer.payload).toEqual(new Uint8Array([0, 0]));
	});

	it('reset() keeps the graph-store doc-change callback wired to the fresh doc', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		let fired = 0;
		client.onDocChange((txn) => {
			if (txn.changed.size > 0) fired++;
		});
		client.start();
		client.reset();
		// A transaction on the FRESH doc still notifies the store (the observer was re-attached).
		seedClientNode(client);
		expect(fired).toBeGreaterThan(0);
	});

	it('synced is false until the manager\'s update lands, and an SV alone does not set it', () => {
		// The flag answers "has this replica pulled from the manager yet" — the automation façade's
		// door onto it (`query.docSynced`) is how a driver knows reads are trustworthy. An inbound
		// SV is the manager ASKING, not answering; only an applied update flips it.
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();
		expect(client.synced).toBe(false);
		ctl.emitSync(syncHello(new Y.Doc())); // manager advertises — still no pull
		expect(client.synced).toBe(false);
		ctl.emitSync(encodeSyncMsg({ kind: 'update', payload: Y.encodeStateAsUpdate(serverWithNode()) }));
		expect(client.synced).toBe(true);
	});

	it('reset() clears synced — the fresh session\'s replica has pulled nothing', () => {
		const ctl = new FakeControl();
		const client = new SyncClient(ctl);
		client.start();
		ctl.emitSync(encodeSyncMsg({ kind: 'update', payload: Y.encodeStateAsUpdate(serverWithNode()) }));
		expect(client.synced).toBe(true);
		client.reset();
		expect(client.synced).toBe(false);
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
