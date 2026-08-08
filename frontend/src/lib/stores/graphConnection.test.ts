import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';

/**
 * The header speaks about the connection ONLY when it needs attention.
 *
 * "Connected" is not news — it is the case in every screenshot of a working app — so the healthy
 * state says nothing at all and takes no width. What the store has to answer is therefore not
 * `connected` but the ALARM: a connection that was established and then lost.
 *
 * The distinction is what makes the boot quiet. `connected` starts false and stays false for the
 * few hundred ms it takes the socket to open, so a bare `!connected` would alarm on every single
 * page load before settling — an alarm the user learns to ignore, which is worse than no alarm.
 */
describe('the connection alarm', () => {
	it('stays quiet at boot, before the socket has ever opened', () => {
		const fc = new FakeControl({ connected: false });
		const g = new GraphStore(fc);
		expect(g.connected, 'the socket has not opened yet').toBe(false);
		expect(g.disconnected, '…and that is the normal boot, not a fault').toBe(false);
	});

	it('stays quiet while connected', () => {
		const g = new GraphStore(new FakeControl());
		expect(g.connected).toBe(true);
		expect(g.disconnected).toBe(false);
	});

	it('alarms once an ESTABLISHED connection is lost, and clears when it comes back', () => {
		const fc = new FakeControl({ connected: false });
		const g = new GraphStore(fc);
		fc.setConnected(true);
		expect(g.disconnected, 'the first connect is not an alarm').toBe(false);
		fc.setConnected(false);
		expect(g.disconnected, 'a socket that WAS there and is not is').toBe(true);
		fc.setConnected(true);
		expect(g.disconnected, 'and the reconnect clears it').toBe(false);
	});

	/* The alarm latches on the first connect, never off it: a second drop after a reconnect is the
	   same fault as the first, and a `connected → disconnected → connected → disconnected` tab must
	   not go quiet on the third transition. */
	it('alarms on every later drop, not just the first', () => {
		const fc = new FakeControl({ connected: false });
		const g = new GraphStore(fc);
		fc.setConnected(true);
		fc.setConnected(false);
		fc.setConnected(true);
		fc.setConnected(false);
		expect(g.disconnected).toBe(true);
	});
});
