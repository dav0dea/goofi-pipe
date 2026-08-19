import { describe, it, expect, vi } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { SyncClient } from './syncClient';
import { nodeView } from './graphDoc';

const OSC = { type: 'Oscillator', name: 'osc', pos: { x: 0, y: 0 } };

/** A document as the manager sends it, whole. */
const stateWith = (nodes: Record<string, unknown>) => ({
	nodes,
	links: [],
	instances: {},
	globals: {},
	arrangement: {}
});

function started(): { ctl: FakeControl; client: SyncClient } {
	const ctl = new FakeControl();
	const client = new SyncClient(ctl);
	client.start();
	return { ctl, client };
}

describe('SyncClient', () => {
	it('is seeded by the doc_state the manager sends unprompted — it asks for nothing', () => {
		const { ctl, client } = started();
		expect(client.synced).toBe(false);
		expect(ctl.recordedCalls().length, 'a replica never speaks: it is read-only').toBe(0);

		ctl.emit({ event: 'doc_state', payload: { v: 7, doc: stateWith({ '1': OSC }) } });
		expect(client.synced).toBe(true);
		expect(client.version).toBe(7);
		expect(nodeView(client.doc, '1')).toMatchObject({ type: 'Oscillator', name: 'osc' });
	});

	it('applies a delta onto the version it names', () => {
		const { ctl, client } = started();
		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': OSC }) } });
		ctl.emit({
			event: 'doc_patch',
			payload: { from: 1, v: 2, patch: { nodes: { '2': { type: 'Buffer', name: 'buf' } } } }
		});
		expect(client.version).toBe(2);
		expect(nodeView(client.doc, '2')).toMatchObject({ type: 'Buffer', name: 'buf' });
		expect(nodeView(client.doc, '1'), 'and leaves the untouched node alone').not.toBeNull();
	});

	it('a null in a delta REMOVES the key — how a merge patch spells a delete', () => {
		// The half a delta format is easiest to get wrong. A replica that merged nulls as values
		// would keep every removed node for ever and look perfectly healthy doing it.
		const { ctl, client } = started();
		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': OSC }) } });
		ctl.emit({ event: 'doc_patch', payload: { from: 1, v: 2, patch: { nodes: { '1': null } } } });
		expect(nodeView(client.doc, '1')).toBeNull();
	});

	it('skips a stale delta in silence — the seed already carried it', () => {
		// The manager subscribes a socket BEFORE it snapshots the document, so a peer's edit landing
		// in that window is broadcast and then included in the snapshot too. Re-delivery is the
		// price of never losing one; a replica that treated it as a gap would stall on every
		// connection made while someone else was editing.
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const { ctl, client } = started();
		ctl.emit({ event: 'doc_state', payload: { v: 5, doc: stateWith({ '1': OSC }) } });
		ctl.emit({ event: 'doc_patch', payload: { from: 3, v: 4, patch: { nodes: { '1': null } } } });
		expect(nodeView(client.doc, '1'), 'the already-applied delete was not replayed').not.toBeNull();
		expect(client.version).toBe(5);
		expect(warn, 'and it is not worth a word').not.toHaveBeenCalled();
		warn.mockRestore();
	});

	it('refuses a delta that reaches PAST this replica, and waits for a fresh doc_state', () => {
		// A gap can only mean the client fell behind the broadcast ring. Applying anyway would leave
		// a replica that looks healthy and is wrong, so it stops until the manager re-seeds it —
		// which the manager does on exactly that lag.
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const { ctl, client } = started();
		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': OSC }) } });
		ctl.emit({
			event: 'doc_patch',
			payload: { from: 5, v: 6, patch: { nodes: { '9': { type: 'Buffer', name: 'b' } } } }
		});
		expect(nodeView(client.doc, '9'), 'the out-of-order delta was not applied').toBeNull();
		expect(client.version, 'and the replica did not move').toBe(1);
		expect(warn).toHaveBeenCalled();

		ctl.emit({ event: 'doc_state', payload: { v: 6, doc: stateWith({ '9': OSC }) } });
		expect(client.version).toBe(6);
		expect(nodeView(client.doc, '9'), 'the re-seed healed it').not.toBeNull();
		warn.mockRestore();
	});

	it('reset() empties the replica so a new session cannot read the old one', () => {
		// A fresh engine mints uids from 1 again, so a surviving document would collide on reused
		// uids and its leaves would read as the new session's.
		const { ctl, client } = started();
		ctl.emit({ event: 'doc_state', payload: { v: 3, doc: stateWith({ '1': OSC }) } });
		client.reset();
		expect(client.synced).toBe(false);
		expect(nodeView(client.doc, '1')).toBeNull();

		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': { type: 'Buffer', name: 'b' } }) } });
		expect(nodeView(client.doc, '1')).toMatchObject({ type: 'Buffer' });
	});

	it('fires the change callback on a seed, on a delta and on a reset', () => {
		const { ctl, client } = started();
		let changes = 0;
		client.onDocChange(() => (changes += 1));
		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': OSC }) } });
		ctl.emit({ event: 'doc_patch', payload: { from: 1, v: 2, patch: { nodes: { '1': { name: 'renamed' } } } } });
		client.reset();
		expect(changes).toBe(3);
	});

	it('stop() unsubscribes, so a later event no longer moves the replica', () => {
		const { ctl, client } = started();
		client.stop();
		ctl.emit({ event: 'doc_state', payload: { v: 1, doc: stateWith({ '1': OSC }) } });
		expect(client.synced).toBe(false);
	});
});
