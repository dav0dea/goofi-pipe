import { describe, it, expect, beforeEach } from 'vitest';
import { workspace } from './workspace.svelte';
import { history } from '$lib/stores/history.svelte';

/**
 * The dirty taxonomy (R spec §4 / D-R3). Every layout mutation replaces `ws.state` and rides the
 * same debounced `set_layout` push, so the store — not the device, not the caller's platform — is
 * where a write is classified as *authoring* (the user edited the arrangement) or *navigation*
 * (the user changed what they are looking at, or the manager echoed its own layout back).
 * `takeLayoutIntent()` is what AppShell hands the manager with each push.
 */
describe('layout write intent', () => {
	beforeEach(() => {
		workspace().reset();
		history().reset();
		workspace().takeLayoutIntent(); // drop whatever the reset itself classified
	});

	it('starts at navigation — nothing has been authored yet', () => {
		expect(workspace().takeLayoutIntent()).toBe('navigation');
	});

	it('classifies a panel split as authoring', () => {
		const ws = workspace();
		ws.split(ws.activePanelId!, 'row');
		expect(ws.takeLayoutIntent()).toBe('authored');
	});

	it('does not classify a REFUSED structural change as authoring', () => {
		const ws = workspace();
		ws.close(ws.activePanelId!); // the last panel cannot be closed
		expect(ws.takeLayoutIntent()).toBe('navigation');
	});

	it('classifies a panel-state write as authoring by default — a viewer kind, a slot pick', () => {
		const ws = workspace();
		ws.setPanelState(ws.activePanelId!, { node: 'osc0', kind: 'image' });
		expect(ws.takeLayoutIntent()).toBe('authored');
	});

	it('classifies an explicitly navigational panel-state write as navigation', () => {
		const ws = workspace();
		ws.setPanelState(ws.activePanelId!, { subpatchPath: '/inst0' }, 'navigation');
		expect(ws.takeLayoutIntent()).toBe('navigation');
	});

	// D-R11: switching layout tabs changes which arrangement is in front, not what any panel
	// holds — the same "looking elsewhere" as entering a sub-patch, and the move `navContext`
	// makes to re-orient an undo.
	it('classifies switching layout tabs as navigation', () => {
		const ws = workspace();
		const first = ws.state.activeWorkspaceId;
		ws.addTab();
		ws.takeLayoutIntent(); // adding the tab IS authoring; isolate the switch
		ws.selectTab(first);
		expect(ws.state.activeWorkspaceId).toBe(first);
		expect(ws.takeLayoutIntent()).toBe('navigation');
	});

	// The manager's own layout coming back on `hello`/`graph_replaced` is not a user write at all.
	// Classifying it as authoring is what re-dirtied a patch moments after it was saved.
	it('classifies the manager echoing a layout back as navigation', () => {
		const ws = workspace();
		const snapshot = ws.serialize();
		ws.split(ws.activePanelId!, 'row');
		ws.takeLayoutIntent();
		ws.hydrate(snapshot);
		expect(ws.takeLayoutIntent()).toBe('navigation');
	});

	it('lets authoring win a mixed debounce window, whichever order it lands in', () => {
		const ws = workspace();
		ws.setPanelState(ws.activePanelId!, { subpatchPath: '/inst0' }, 'navigation');
		ws.split(ws.activePanelId!, 'row');
		expect(ws.takeLayoutIntent()).toBe('authored');

		ws.split(ws.activePanelId!, 'column');
		ws.setPanelState(ws.activePanelId!, { subpatchPath: '/' }, 'navigation');
		expect(ws.takeLayoutIntent()).toBe('authored');
	});

	it('resets on take, so one authored write cannot dirty every later push', () => {
		const ws = workspace();
		ws.split(ws.activePanelId!, 'row');
		expect(ws.takeLayoutIntent()).toBe('authored');
		ws.setPanelState(ws.activePanelId!, { subpatchPath: '/inst0' }, 'navigation');
		expect(ws.takeLayoutIntent()).toBe('navigation');
	});
});
