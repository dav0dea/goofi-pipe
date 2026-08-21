/** Canvas colour palette: canvas 2D/WebGL cannot read the CSS custom properties. */

export const SERIES: string[] = [
	'#7ab7ff',
	'#b58cff',
	'#5dd09a',
	'#ffb761',
	'#ff7aa2',
	'#9aa3b3',
	'#c5c8d6',
	'#6e7686'
];

export const AXIS_INK = 'rgba(208, 208, 208, 0.55)';

/** The app's mono stack at `px`, ready for `ctx.font` — spelled out, since a canvas
 * context cannot resolve `var(--font-mono)` and discards the whole declaration. */
export function tickFont(px: number): string {
	return `${px}px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
}
