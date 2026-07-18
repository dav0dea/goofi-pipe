import { describe, it, expect } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { globalsMap } from '$lib/crdt/graphDoc';
import * as Y from 'yjs';

/** Seed a system global into the store's doc exactly as the manager's mirror writes it (in one
 * transaction so the store's afterTransaction → _syncFromDoc derives `g.globals` once). */
function seedSystemGlobal(g: GraphStore, name: string, value: number): void {
	Y.transact(g.doc, () => {
		const e = new Y.Map<unknown>();
		e.set('value', value);
		e.set('type', 'float');
		e.set('system', true);
		globalsMap(g.doc).set(name, e);
	});
}

describe('GraphStore globals mutators — the seam the panel + agent drive', () => {
	it('adds a user global and reflects it in the reactive list (+ broadcasts a delta)', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const before = fc.sentSyncFrames.length;

		expect(g.addGlobal('gain', 2.5, 'float')).toBe(true);
		expect(g.globals).toEqual([{ name: 'gain', value: 2.5, type: 'float', system: false }]);
		// The write was broadcast to the manager (a client leaf-write), like any doc edit.
		expect(fc.sentSyncFrames.length).toBeGreaterThan(before);

		// A duplicate / invalid name does not land and derives nothing new.
		expect(g.addGlobal('gain', 9, 'float')).toBe(false);
		expect(g.addGlobal('1bad', 0, 'int')).toBe(false);
		expect(g.globals).toHaveLength(1);
	});

	it('edits an existing value, keeping type + system', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seedSystemGlobal(g, 'default_ufreq', 30);
		expect(g.globals[0]).toEqual({ name: 'default_ufreq', value: 30, type: 'float', system: true });

		expect(g.setGlobalValue('default_ufreq', 45)).toBe(true);
		expect(g.globals[0]).toEqual({ name: 'default_ufreq', value: 45, type: 'float', system: true });
	});

	it('removes a user global but refuses a system one', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seedSystemGlobal(g, 'default_ufreq', 30);
		g.addGlobal('subject', 'P07', 'string');

		expect(g.removeGlobal('subject')).toBe(true);
		expect(g.globals.map((v) => v.name)).toEqual(['default_ufreq']);
		// System global is protected client-side (the manager also rejects it).
		expect(g.removeGlobal('default_ufreq')).toBe(false);
		expect(g.globals.map((v) => v.name)).toEqual(['default_ufreq']);
	});

	it('reports landed (true) for an idempotent value set / same-name rename', () => {
		// The writer's "landed" contract, not commit()'s delta flag: an at-target edit lands even
		// though it broadcasts nothing — the agent façade keys false on invalid/collision/system only.
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seedSystemGlobal(g, 'default_ufreq', 30);
		g.addGlobal('gain', 2, 'float');

		expect(g.setGlobalValue('default_ufreq', 30)).toBe(true); // unchanged value still landed
		expect(g.renameGlobal('gain', 'gain')).toBe(true); // same-name rename is a no-op that landed
		// A genuine failure still reports false.
		expect(g.setGlobalValue('ghost', 1)).toBe(false);
	});

	it('renames a user global (carrying value+type), refusing system + collisions', () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seedSystemGlobal(g, 'default_ufreq', 30);
		g.addGlobal('gain', 2, 'float');

		expect(g.renameGlobal('gain', 'gain_a')).toBe(true);
		expect(g.globals.find((v) => v.name === 'gain_a')).toEqual({
			name: 'gain_a',
			value: 2,
			type: 'float',
			system: false
		});
		expect(g.renameGlobal('default_ufreq', 'x')).toBe(false); // system, protected
		expect(g.globals.some((v) => v.name === 'default_ufreq')).toBe(true);
	});
});
