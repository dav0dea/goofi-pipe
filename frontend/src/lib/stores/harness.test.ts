import { describe, expect, it, vi } from 'vitest';
import { FakeControl } from '$lib/test/fakeControl';
import { HarnessStore, type HarnessRoster } from './harness.svelte';
import { endTermSession, liveTermSessions, termSession, type TerminalLike } from './termSession';
import { notify } from './notify.svelte';
import type { GraphSnapshot } from '$lib/api/control';

/** A terminal that needs no DOM — the session store's own test drives the real behaviour. */
const fakeTerm = (): TerminalLike => ({
	element: undefined,
	open: () => {},
	write: () => {},
	resize: () => {},
	onData: () => ({ dispose: () => {} }),
	proposeDimensions: () => undefined,
	dispose: () => {}
});

type Row = [string, HarnessRoster['instances'][number]['state'], number?];

const roster = (...ids: Row[]): HarnessRoster => ({
	instances: ids.map(([id, state, exit_code]) => ({ id, harness: '_sh', state, exit_code })),
	detected: [{ harness: 'claude', path: '/usr/bin/claude', version: 'claude 1.2.3' }]
});

const hello = (h: HarnessRoster) =>
	({
		event: 'hello',
		payload: { harnesses: h } as unknown as GraphSnapshot
	}) as const;

describe('the harness roster', () => {
	it('is seeded by the snapshot and replaced by every harness_changed', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		expect(h.instances).toEqual([]);

		ctl.emit(hello(roster(['a', 'running'])));
		expect(h.instances.map((i) => i.id)).toEqual(['a']);
		expect(h.detected[0].harness).toBe('claude');

		// A dead harness is not part of the app's roster: it has no interactivity and no state left,
		// so it is dropped rather than kept as a frozen screen (user, 2026-08-10).
		ctl.emit({ event: 'harness_changed', payload: roster(['a', 'exited', 0], ['b', 'running']) });
		expect(h.instances.map((i) => i.id)).toEqual(['b']);
		expect(h.running).toBe(1);
	});

	it('takes the panel of a harness that died back to its launcher, terminal and all', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'])));
		vi.stubGlobal(
			'WebSocket',
			class {
				binaryType = '';
				readyState = 3;
				addEventListener(): void {}
				send(): void {}
				close(): void {}
			}
		);
		vi.stubGlobal('location', { protocol: 'http:', host: 'h' });
		termSession('a', fakeTerm);
		h.mount('panel-1');
		expect(h.instanceFor('panel-1')).toBe('a');

		ctl.emit({ event: 'harness_changed', payload: roster(['a', 'exited', 0]) });
		expect(h.instanceFor('panel-1'), 'the panel is back on its launcher').toBe(null);
		expect(liveTermSessions(), 'and the dead terminal is gone with it').toEqual([]);
		// …and the manager is told to drop it, so its roster and this one agree about what exists.
		expect(ctl.recordedCalls().at(-1)).toEqual({
			op: 'agent stop',
			payload: { instance: 'a' }
		});
		vi.unstubAllGlobals();
	});

	it('says a crash out loud, and says nothing about a clean exit or a kill', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'], ['b', 'running'], ['c', 'running'])));
		expect(h.running).toBe(3);

		// Nobody asked it to stop and it did not end well: the one case worth interrupting for.
		notify().clear();
		ctl.emit({ event: 'harness_changed', payload: roster(['a', 'exited', 3], ['b', 'running'], ['c', 'running']) });
		expect(notify().message).toMatch(/_sh.*3/);

		// A clean exit is the user typing `exit` — an ordinary end, not an alarm.
		notify().clear();
		ctl.emit({ event: 'harness_changed', payload: roster(['b', 'exited', 0], ['c', 'running']) });
		expect(notify().message).toBe(null);

		// A kill was asked for, so its death is not news. `stopping` is what carries that: the
		// manager broadcasts it the moment the stop is asked for, a whole grace before the child goes.
		ctl.emit({ event: 'harness_changed', payload: roster(['c', 'stopping']) });
		ctl.emit({ event: 'harness_changed', payload: roster(['c', 'exited', 143]) });
		expect(notify().message).toBe(null);
		expect(h.running).toBe(0);
	});

	it('gives each panel its own instance, and never takes one another panel is showing', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'], ['b', 'running'])));

		h.mount('panel-1');
		expect(h.instanceFor('panel-1')).toBe('a');
		h.mount('panel-2');
		expect(h.instanceFor('panel-2'), 'the second panel takes the free one').toBe('b');
		expect(h.instanceFor('panel-1'), 'and does not steal the first').toBe('a');

		// A panel with nothing free shows its empty state rather than fighting for a terminal.
		h.mount('panel-3');
		expect(h.instanceFor('panel-3')).toBe(null);
		// …but the switcher is an explicit choice, so it wins.
		h.show('panel-3', 'a');
		expect(h.instanceFor('panel-3')).toBe('a');

		// Unmounting frees the binding; the SESSION (and its scrollback) is untouched by it, which
		// is what lets the next panel pick the same instance up where this one left it.
		h.unmount('panel-1');
		h.unmount('panel-3');
		expect(h.instanceFor('panel-1')).toBe(null);
		h.mount('panel-4');
		expect(h.instanceFor('panel-4'), 'the freed instance is claimable again').toBe('a');
	});

	it('leaves a detached panel on its launcher instead of re-claiming what it let go of', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'])));
		h.mount('panel-1');
		expect(h.instanceFor('panel-1')).toBe('a');

		h.release('panel-1');
		expect(h.instanceFor('panel-1')).toBe(null);
		expect(h.closing, 'answering the question closes it').toBe(null);
		// `a` is free again, and this panel is the only one — so the auto-claim WOULD take it back on
		// the next roster event, which is every detection sweep. Letting go has to be remembered.
		ctl.emit({ event: 'harness_changed', payload: roster(['a', 'running']) });
		expect(h.instanceFor('panel-1')).toBe(null);
		// A panel that never chose still claims, so a harness spawned from anywhere lands somewhere.
		h.mount('panel-2');
		expect(h.instanceFor('panel-2')).toBe('a');
	});

	it('never writes the panel choice to the manager — it is viewpoint, not layout', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'], ['b', 'running'])));
		h.mount('panel-1');
		h.show('panel-1', 'b');
		h.unmount('panel-1');
		// Not "no edit_panel" — NOTHING. A choice that reached the manager at all would ride the
		// `.gfi`, converge to peers and enter undo, which is what makes this a taxonomy question
		// (CLAUDE.md: navigation must not dirty) rather than a preference.
		expect(ctl.recordedCalls()).toEqual([]);
	});

	it('ends a session when its instance leaves the roster', () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster(['a', 'running'])));
		vi.stubGlobal('WebSocket', class {
			binaryType = '';
			readyState = 3;
			addEventListener(): void {}
			send(): void {}
			close(): void {}
		});
		vi.stubGlobal('location', { protocol: 'http:', host: 'h' });
		termSession('a', fakeTerm);
		h.mount('panel-1');
		expect(liveTermSessions()).toEqual(['a']);

		// A departure — the patch was torn down, or another tab dismissed it. The terminal goes with
		// it, and the panel it was showing in lets go.
		ctl.emit({ event: 'harness_changed', payload: roster() });
		expect(liveTermSessions()).toEqual([]);
		expect(h.instanceFor('panel-1')).toBe(null);
		vi.unstubAllGlobals();
		endTermSession('a');
	});

	it('launches into the panel that asked, and kills through the manager', async () => {
		const ctl = new FakeControl();
		const h = new HarnessStore(ctl);
		ctl.emit(hello(roster()));
		ctl.setCallResult('agent start', { instance_id: 'fresh' });

		h.mount('panel-1');
		await h.launch('panel-1', 'claude');
		expect(ctl.recordedCalls()).toEqual([{ op: 'agent start', payload: { name: 'claude' } }]);
		// The roster arrives on its own event; the binding is what the launch decided.
		ctl.emit({ event: 'harness_changed', payload: roster(['fresh', 'running']) });
		expect(h.instanceFor('panel-1')).toBe('fresh');

		h.kill('fresh');
		expect(ctl.recordedCalls().at(-1)).toEqual({
			op: 'agent stop',
			payload: { instance: 'fresh' }
		});
	});
});
