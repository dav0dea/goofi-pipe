/** Pure device arithmetic. No DOM — vitest runs in node with no visualViewport, so the testable
 * logic lives here; device.svelte.ts wires it to the browser.
 *
 * There is exactly one function, deliberately (D-R8): the size/pointer classes this module used to
 * compute were write-only, and `@media` answers them natively. The keyboard overlap is the one
 * device fact CSS cannot see at all. */

/** Pixels the soft keyboard overlaps the layout viewport (never negative). */
export function kbInset(viewportHeight: number, innerHeight: number): number {
	return Math.max(0, innerHeight - viewportHeight);
}
