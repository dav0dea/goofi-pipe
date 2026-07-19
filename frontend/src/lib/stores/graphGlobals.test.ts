import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { GraphStore } from './graph.svelte';
import { globalsMap } from '$lib/crdt/graphDoc';
import { history } from './history.svelte';
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

// Globals are now server COMMAND ops (EditGlobal / a Compound rename), undoable and validated
// server-side (invalid name / collision / protected-system reject the RPC — covered by the bridge +
// engine tests). This file pins the CLIENT's job: map each mutator to the right command op/payload
// and record an undoable step, propagating a server rejection.
describe('GraphStore globals mutators — the command surface the panel + agent drive', () => {
	beforeEach(() => history().reset());
	const found = (fc: FakeControl, op: string) => fc.recordedCalls().find((c) => c.op === op)?.payload;

	it('addGlobal issues set_global{name,value,type} and records an undoable step', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		await g.addGlobal('gain', 2.5, 'float');
		expect(found(fc, 'set_global')).toEqual({ name: 'gain', value: 2.5, type: 'float' });
		expect(history().canUndo).toBe(true);
	});

	it('setGlobalValue looks up the declared type and issues set_global', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seedSystemGlobal(g, 'default_ufreq', 30);
		await g.setGlobalValue('default_ufreq', 45);
		expect(found(fc, 'set_global')).toEqual({ name: 'default_ufreq', value: 45, type: 'float' });
	});

	it('setGlobalValue rejects an unknown global (no command sent)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		await expect(g.setGlobalValue('ghost', 1)).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'set_global')).toBe(false);
	});

	it('removeGlobal issues remove_global{name}', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		await g.removeGlobal('subject');
		expect(found(fc, 'remove_global')).toEqual({ name: 'subject' });
	});

	it('renameGlobal issues rename_global{old,new} as one undoable step', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		await g.renameGlobal('gain', 'gain_a');
		expect(found(fc, 'rename_global')).toEqual({ old: 'gain', new: 'gain_a' });
		expect(history().canUndo).toBe(true);
	});

	it('a server rejection propagates (name/collision/system are validated server-side)', async () => {
		const fc = new FakeControl();
		fc.failNext('set_global');
		const g = new GraphStore(fc);
		await expect(g.addGlobal('1bad', 0, 'int')).rejects.toThrow();
		// A rejected edit records no undo step.
		expect(history().canUndo).toBe(false);
	});
});
