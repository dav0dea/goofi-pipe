/**
 * View-domain executor: undo/redo of a data viewer's kind + cog-menu settings.
 * Viewer view-state is frontend-only, so this just re-applies a captured
 * snapshot to wherever the viewer's state lives — the node-scoped inline store
 * (node body viewer) or a docked Viewer panel's layout state.
 */
import { setInlineFullView } from './inlineView.svelte';
import { history } from '$lib/stores/history.svelte';
import { captureNavContext } from '$lib/workspace/navContext';
import type { SettingsMap } from './settingsSchema';
import type { Executor, ExecutorDeps, ViewAction, ViewSnapshot } from '$lib/stores/history.svelte';

function apply(target: ViewAction['payload']['target'], snap: ViewSnapshot, deps: ExecutorDeps): void {
	setInlineFullView(target.node, target.slot, snap);
	// Round-trip into node.viewers (debounced) like a click would. No-op if the node is gone.
	deps.graph.pushNodeViewers(target.node);
}

const setView: Executor = {
	async forward(action, deps) {
		const a = action as ViewAction;
		apply(a.payload.target, a.payload.after, deps);
	},
	async inverse(action, deps) {
		const a = action as ViewAction;
		apply(a.payload.target, a.payload.before, deps);
	}
};

export const viewExecutors: Record<string, Executor> = { set_view: setView };

/** Record a viewer kind/settings change as an undoable `set_view` action.
 * Called by both binding setters AFTER they apply the change, with the raw
 * (pre-resolution) before/after snapshots. No-op while a replay is suspended,
 * and skips no-op changes. Imports history lazily to avoid an import cycle. */
export function recordViewChange(
	target: ViewAction['payload']['target'],
	before: ViewSnapshot,
	after: ViewSnapshot,
	label: string
): void {
	if (before.kind === after.kind && shallowEqual(before.settings, after.settings)) return;
	if (history().isSuspended) return;
	history().record({
		kind: 'set_view',
		domain: 'view',
		label,
		context: captureNavContext(),
		payload: { target, before, after }
	});
}

function shallowEqual(a: SettingsMap, b: SettingsMap): boolean {
	const ak = Object.keys(a);
	const bk = Object.keys(b);
	if (ak.length !== bk.length) return false;
	return ak.every((k) => a[k] === b[k]);
}
