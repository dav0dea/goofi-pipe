/**
 * Post-undo/redo highlight pulse (backlog #19). After a replay reorients to the
 * affected nodes, briefly flash them. A node already on the canvas flashes at
 * once; one being re-created by the inverse/forward (e.g. undo of a delete) is
 * flashed only once its `node_added` echo lands — so the pulse starts when the
 * node actually appears, via `awaitEvent`, rather than racing the backend.
 */
import { awaitEvent } from '$lib/api/awaitEvent';
import { flash } from './flash.svelte';
import type { ControlEvent } from '$lib/api/control';
import type { ExecutorDeps, NavContext } from './history.svelte';

/** Window to wait for a re-created node to echo before giving up on its flash. */
const ECHO_TIMEOUT_MS = 2000;

export function pulseRestored(ctx: NavContext, deps: ExecutorDeps): void {
	const names = new Set<string>();
	for (const s of Object.values(ctx.selection)) for (const n of s.nodes) names.add(n);
	if (!names.size) return;

	for (const name of names) {
		if (deps.graph.nodeByName(name)) {
			flash().pulse([name]);
			continue;
		}
		// Re-created by the replay — flash when its node_added echo arrives.
		void awaitEvent<ControlEvent>(
			(cb) => deps.control.on(cb),
			(ev) => ev.event === 'node_added' && (ev.payload as { name?: string }).name === name,
			ECHO_TIMEOUT_MS
		)
			.then(() => flash().pulse([name]))
			.catch(() => {
				/* node never reappeared within the window — no flash */
			});
	}
}
