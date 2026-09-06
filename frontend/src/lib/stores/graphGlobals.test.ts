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
		await g.addGlobal('patch.gain', 2.5, 'float');
		expect(found(fc, 'global entry add')).toEqual({ name: 'patch.gain', value: 2.5, type: 'float' });
		expect(history().canUndo).toBe(true);
	});

	it('addGlobal refuses a name already taken — set_global would overwrite it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.global('patch.gain', { value: 1, type: 'float', system: false });
		await expect(g.addGlobal('patch.gain', 2.5, 'float')).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'global entry add')).toBe(false);
	});

	it('setGlobalValue issues global edit — the held type stays, unsent', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		seedSystemGlobal(d, 'system.default_ufreq', 30);
		await g.setGlobalValue('system.default_ufreq', 45);
		expect(found(fc, 'global entry edit')).toEqual({ name: 'system.default_ufreq', value: 45 });
	});

	it('setGlobalValue rejects an unknown global (no command sent)', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		await expect(g.setGlobalValue('ghost', 1)).rejects.toThrow();
		expect(fc.recordedCalls().some((c) => c.op === 'global entry edit')).toBe(false);
	});

	it('removeGlobal issues global remove', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		await g.removeGlobal('patch.subject');
		expect(found(fc, 'global entry remove')).toEqual({ name: 'patch.subject' });
	});

	it('renameGlobal is ONE op, and the manager rewrites every expression reading it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.global('patch.gain', { value: 2.5, type: 'float', system: false });
		await g.renameGlobal('patch.gain', 'patch.level');
		expect(found(fc, 'global entry rename')).toEqual({ name: 'patch.gain', to: 'patch.level' });
		expect(history().canUndo).toBe(true);
	});

	it('renameGlobalGroup is ONE op too, and moves every member with it', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		const d = seed(fc);
		d.global('patch.gain', { value: 2.5, type: 'float', system: false });
		await g.renameGlobalGroup('patch', 'desk');
		expect(found(fc, 'global group rename')).toEqual({ from: 'patch', to: 'desk' });
		expect(history().canUndo).toBe(true);
	});

	it('a lock is ONE op on the entry or on the group, naming only the axis it turns', async () => {
		const fc = new FakeControl();
		const g = new GraphStore(fc);
		seed(fc).global('patch.gain', { value: 2.5, type: 'float' });
		await g.lockGlobal('patch.gain', { value: true });
		expect(found(fc, 'global entry lock')).toEqual({ name: 'patch.gain', value: true });
		await g.lockGlobalGroup('patch', { config: true });
		expect(found(fc, 'global group lock')).toEqual({ group: 'patch', config: true });
		expect(history().canUndo).toBe(true);
	});

	it('a server rejection propagates (name/collision/system are validated server-side)', async () => {
		const fc = new FakeControl();
		fc.failNext('global entry add');
		const g = new GraphStore(fc);
		const d = seed(fc);
		await expect(g.addGlobal('bad', 0, 'int')).rejects.toThrow();
		// A rejected add records no undo step.
		expect(history().canUndo).toBe(false);
	});
});
