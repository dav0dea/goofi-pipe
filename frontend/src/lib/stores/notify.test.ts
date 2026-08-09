import { describe, it, expect } from 'vitest';
import { NotifyStore } from './notify.svelte';

describe('NotifyStore — the app-wide transient alarm channel', () => {
	it('starts silent', () => {
		expect(new NotifyStore().message).toBe(null);
	});

	it('raise() publishes the message, clear() takes it back down', () => {
		const n = new NotifyStore();
		n.raise('Save failed: Permission denied');
		expect(n.message).toBe('Save failed: Permission denied');
		n.clear();
		expect(n.message).toBe(null);
	});

	/** The producers (an undo/redo replay, a failed save or load) share ONE channel, so the newest
	 * alarm is what the user reads — a stale one must never sit over it. */
	it('the latest raise wins', () => {
		const n = new NotifyStore();
		n.raise('Undo failed: name taken');
		n.raise('Save failed: No such file or directory');
		expect(n.message).toBe('Save failed: No such file or directory');
	});

	/** `raise` is called from RPC catch blocks, where the thrown value is an `unknown`. */
	it('failure() renders a verb and any thrown value as one line', () => {
		const n = new NotifyStore();
		n.failure('Save', new Error('Permission denied'));
		expect(n.message).toBe('Save failed: Permission denied');
		n.failure('Load', 'not a .gfi archive');
		expect(n.message).toBe('Load failed: not a .gfi archive');
	});
});
