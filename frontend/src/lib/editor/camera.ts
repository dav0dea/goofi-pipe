/**
 * A node editor's camera, kept per PANEL rather than per component instance.
 *
 * The layout tree is rebuilt from the arrangement on every change, so a panel's content component
 * is destroyed and recreated whenever the tree RESHAPES around it — a split turns the leaf above it
 * into a branch, a close collapses one back — and switching page swaps the whole tree. The pan/zoom
 * lives inside <SvelteFlow>, so each of those threw the user's framing away and the on-load fit
 * then re-centred the graph.
 *
 * This is the one owner of that framing. `fittedEpoch` rides with it because it answers the same
 * question: a remount is not a load, so the fit that belongs to a load must not fire again for a
 * graph already framed.
 *
 * Never pruned. A panel id is minted once per session and a closed one is never reissued, so a
 * stale entry is unreachable rather than wrong — and the one id spelling that DOES recur, across a
 * patch load, arrives with a bumped load epoch that re-fits over it.
 */
import type { Viewport } from '@xyflow/svelte';

export interface Camera {
	/** Where the editor is looking. `null` for a panel whose flow has yet to report one. */
	viewport: Viewport | null;
	/** Graph load epoch this camera was framed for; -1 until it has been fitted once. */
	fittedEpoch: number;
}

const cameras = new Map<string, Camera>();

/** This panel's camera, minted empty on the first ask. Mutate it in place — it IS the state. */
export function camera(panelId: string): Camera {
	let c = cameras.get(panelId);
	if (!c) cameras.set(panelId, (c = { viewport: null, fittedEpoch: -1 }));
	return c;
}
