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
