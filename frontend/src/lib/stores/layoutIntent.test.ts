import { describe, it, expect, beforeEach } from 'vitest';
import { workspace, type Workspace } from 'panelty';
import { FakeControl } from '$lib/test/fakeControl';
import { goofiLayoutHost } from './layoutHost';
import { history } from './history.svelte';

/**
 * The dirty taxonomy (R spec §4 / D-R3), after the arrangement became the manager's.
 *
 * It used to be a CLASSIFICATION the client folded and declared (`takeLayoutIntent` →
 * `set_layout {intent}`). Now it is the ROUTING: an authored write leaves as a layout command,
 * which the manager dirties like any other; a navigational one never leaves as one at all — it is
 * viewpoint, and `set_viewpoint` is the one write op with an explicit exception from the dirty
 * gate. The taxonomy is the same and the guarantees are the same; what changed is that they hold by
 * construction rather than by the client getting a flag right. So these tests assert what a write
 * BECOMES, which is the thing the manager acts on.
 *
 * `dirty-taxonomy.spec.ts` pins the same rule end to end, against the real dot.
 */

const LAYOUT_OPS = [
	'place_panel',
	'edit_panel',
	'remove_panel'
];

function defaultTabs(): Workspace[] {
	return [
		{ id: 'tab-1', name: 'Tab 1', root: { kind: 'panel', id: 'panel-2', panelType: 'node-editor' } }
	];
}

let fc: FakeControl;

/** Whether anything sent since boot was a write to the ARRANGEMENT — i.e. dirties the patch. */
function dirtied(): boolean {
	return fc.recordedCalls().some((c) => LAYOUT_OPS.includes(c.op));
}

function boot(tabs: Workspace[] = defaultTabs()): ReturnType<typeof workspace> {
	fc = new FakeControl();
	const ws = workspace();
	ws.configureHost(goofiLayoutHost({ control: () => fc }));
	ws.commitResize('#none');
	ws.syncFromDoc(tabs);
	return ws;
}

describe('layout write intent', () => {
	beforeEach(() => history().reset());

	it('starts clean — nothing has been authored yet', () => {
		boot();
		expect(dirtied()).toBe(false);
	});

	it('makes a panel split an authoring write', async () => {
		const ws = boot();
		ws.split('panel-2', 'row');
		await Promise.resolve();
		expect(dirtied()).toBe(true);
	});

	it('does not let a REFUSED structural change count as an edit', async () => {
		const ws = boot();
		fc.failNext('remove_panel');
		ws.close('panel-2'); // the last panel cannot be closed
		await Promise.resolve();
		await Promise.resolve();
		expect(history().length, 'a refusal is not a step, and the manager never dirtied').toBe(0);
	});

	it('makes a panel-state write authoring by default — a viewer kind, a slot pick', async () => {
		const ws = boot();
		ws.setPanelState('panel-2', { node: 'osc0', kind: 'image' });
		await Promise.resolve();
		expect(dirtied()).toBe(true);
	});

	it('keeps an explicitly navigational panel-state write off the arrangement entirely', async () => {
		const ws = boot();
		ws.setPanelState('panel-2', { subpatchPath: '/inst0' }, 'navigation');
		await Promise.resolve();
		expect(dirtied(), 'entering a sub-patch must not dirty, on any platform').toBe(false);
		expect(fc.recordedCalls(), 'and it is not sent at all — it is this client’s viewpoint').toEqual(
			[]
		);
	});

	// D-R11: switching layout tabs changes which arrangement is in front, not what any panel
	// holds — the same "looking elsewhere" as entering a sub-patch, and the move `navContext`
	// makes to re-orient an undo.
	it('keeps switching layout tabs off the arrangement', async () => {
		const ws = boot([
			...defaultTabs(),
			{ id: 'tab-7', name: 'Second', root: { kind: 'panel', id: 'panel-8', panelType: 'console' } }
		]);
		ws.selectTab('tab-7');
		await Promise.resolve();
		expect(dirtied()).toBe(false);
		expect(ws.state.activeWorkspaceId, 'the tab did switch').toBe('tab-7');
	});

	it('makes creating a tab authoring, and only the SELECTION a look', async () => {
		const ws = boot();
		ws.addTab();
		await Promise.resolve();
		expect(dirtied(), 'creating a tab really does change the patch').toBe(true);
	});

	// The other half of the axis, and the one that used to be a client-declared flag: a viewpoint
	// still PERSISTS. Persistence and dirtiness are separate axes — the shell pushes this through
	// `set_viewpoint`, which the manager stores into the `.gfi` without touching the dot.
	it('still hands the viewpoint over to be persisted', () => {
		const ws = boot();
		ws.setPanelState('panel-2', { subpatchPath: '/inst0' }, 'navigation');
		expect(ws.viewpoint()).toMatchObject({ paths: { 'panel-2': '/inst0' } });
	});
});
