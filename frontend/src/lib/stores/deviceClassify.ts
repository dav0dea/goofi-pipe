/** Pure device-class arithmetic (spec §4.2). No DOM — vitest runs in node with no matchMedia,
 * so all the testable logic lives here; device.svelte.ts wires it to the browser. */

export type SizeClass = 'phone' | 'compact' | 'full';

/** Size class from the viewport box. `phone` if either axis is small; then width bands. */
export function classify(width: number, height: number, _opts?: { coarse?: boolean }): { size: SizeClass; short: boolean } {
	const short = height <= 480;
	let size: SizeClass;
	if (width <= 600 || height <= 480) size = 'phone';
	else if (width < 960) size = 'compact';
	else size = 'full';
	return { size, short };
}

/** Pixels the soft keyboard overlaps the layout (never negative). */
export function kbInset(viewportHeight: number, innerHeight: number): number {
	return Math.max(0, innerHeight - viewportHeight);
}
