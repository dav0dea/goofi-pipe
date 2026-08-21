/** Pure device arithmetic, with no DOM; device.svelte.ts wires it to the browser. */

/** Pixels the soft keyboard overlaps the layout viewport (never negative). */
export function kbInset(viewportHeight: number, innerHeight: number): number {
	return Math.max(0, innerHeight - viewportHeight);
}
