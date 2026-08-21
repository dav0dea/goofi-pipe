/**
 * A node editor's camera, kept per PANEL rather than per component instance, so a layout reshape
 * does not throw the user's framing away. Never pruned: a panel id is never reissued.
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
