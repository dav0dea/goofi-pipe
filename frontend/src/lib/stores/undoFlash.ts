/**
 * Post-undo/redo highlight pulse (backlog #19). After a replay reorients to the
 * affected nodes, briefly flash the ones already on the canvas.
 */
import { flash } from './flash.svelte';
import type { ExecutorDeps, NavContext } from './history.svelte';

export function pulseRestored(ctx: NavContext, deps: ExecutorDeps): void {
	const names = new Set<string>();
	for (const s of Object.values(ctx.selection)) for (const n of s.nodes) names.add(n);
	if (!names.size) return;

	// Nodes a replay *re-creates* are not flashed: undo/redo reach the client as a whole-doc
	// re-mirror, with no per-node event to wait on. (The previous code awaited a `node_added`
	// echo that a replay never emits, so it could only ever time out.)
	for (const name of names) {
		if (deps.graph.nodeById(name)) flash().pulse([name]);
	}
}
