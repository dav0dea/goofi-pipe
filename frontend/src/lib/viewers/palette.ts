/** Canvas colour palette. Canvas 2D/WebGL cannot read CSS custom properties, so the
 * series palette lives here in TS, at one source of truth so its copies can never
 * drift (they did: an 8-vs-7 series copy, audit §5). */

// The canonical multi-series line palette — the set ArrayViewer's uPlot lines use.
// TrajectoryViewer had drifted to a 7-colour copy (missing the last); both now share this.
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

// The ink every viewer draws its canvas chrome in — tick labels and corner readouts. Quiet
// enough to read as a scale annotation rather than as data.
export const AXIS_INK = 'rgba(208, 208, 208, 0.55)';

/** The app's mono stack at `px`, ready for `ctx.font`.
 *
 * The stack has to be spelled out: a canvas context parses `font` as a standalone CSS value with
 * no element to resolve custom properties against, so `ctx.font = '11px var(--font-mono)'` is
 * silently discarded and the context keeps the UA's 10px sans-serif. Mirrors app.css's
 * `--font-mono` — the one restatement, here, instead of a shortened copy per viewer.
 */
export function tickFont(px: number): string {
	return `${px}px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
}
