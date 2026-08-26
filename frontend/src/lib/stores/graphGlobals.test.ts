import { describe, it, expect, beforeEach } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { seed, type DocSeed } from '$lib/test/docSeed';
import { GraphStore } from './graph.svelte';
import { history } from './history.svelte';

/** Seed a system global the way the manager sends one. */
function seedSystemGlobal(d: DocSeed, name: string, value: number): void {
	d.global(name, { value, type: 'float', system: true });
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
		const d = seed(fc);
		await g.addGlobal('gain', 2.5, 'float');
		expect(found(fc, 'global add')).toEqual({ name: 'gain', value: 2.5, type: 'float' });
		expect(history().canUndo).toBe(true);
	});

	it('addGlobal refuses a name already taken — set_global would overwrite it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.global('gain', { value: 1, type: 'float', system: false });
		await expect(g.addGlobal('gain', 2.5, 'float')).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'global add')).toBe(false);
	});

	it('setGlobalValue issues global edit — the held type stays, unsent', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		seedSystemGlobal(d, 'default_ufreq', 30);
		await g.setGlobalValue('default_ufreq', 45);
		expect(found(fc, 'global edit')).toEqual({ name: 'default_ufreq', value: 45 });
	});

	it('setGlobalValue rejects an unknown global (no command sent)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		await expect(g.setGlobalValue('ghost', 1)).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'global edit')).toBe(false);
	});

	it('removeGlobal issues global remove', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		await g.removeGlobal('subject');
		expect(found(fc, 'global remove')).toEqual({ name: 'subject' });
	});

	it('renameGlobal compounds the set and the delete into one undoable step', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.global('gain', { value: 2.5, type: 'float', system: false });
		await g.renameGlobal('gain', 'gain_a');
		expect(found(fc, 'compound')).toEqual({
			ops: [
				{ op: 'global add', payload: { name: 'gain_a', value: 2.5, type: 'float' } },
				{ op: 'global remove', payload: { name: 'gain' } }
			]
		});
		expect(history().canUndo).toBe(true);
	});

	it('a server rejection propagates (name/collision/system are validated server-side)', async () => {
		const fc = new FakeControl();
		fc.failNext('global add');
		const g = new GraphStore(fc);
		const d = seed(fc);
		await expect(g.addGlobal('1bad', 0, 'int')).rejects.toThrow();
		// A rejected add records no undo step.
		expect(history().canUndo).toBe(false);
	});
});
