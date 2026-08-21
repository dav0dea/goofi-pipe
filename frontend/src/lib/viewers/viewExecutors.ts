/**
 * Undo/redo of an INLINE viewer's kind + settings, which no manager command owns —
 * a docked panel's is layout state and never comes here.
 */
import { history } from '$lib/stores/history.svelte';
import { captureNavContext } from '$lib/stores/navContext';
import type { SettingsMap } from './settingsSchema';
import type { Executor, ExecutorDeps, ViewAction, ViewSnapshot } from '$lib/stores/history.svelte';

function apply(target: ViewAction['payload']['target'], snap: ViewSnapshot, deps: ExecutorDeps): void {
	// A snapshot carries no collapse, so an undo leaves the viewer open or shut as the user has it.
	deps.graph.setSlotView(target.node, target.slot, snap);
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

/** Record a viewer kind/settings change as an undoable `set_view` action, called by the
 * binding setters AFTER they apply it with the raw (pre-resolution) snapshots. */
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
