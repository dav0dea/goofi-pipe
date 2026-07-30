/**
 * The numpy surface offered after `np.`.
 *
 * **CURATED, NOT AUTHORITATIVE.** This is a hand-picked list of the numpy names an expression
 * realistically reaches for, not `dir(np)`. The real thing is reachable — the repo already runs a
 * live interpreter for node discovery (`crates/goofi-node/src/discover.rs`) and could answer a
 * `complete_expr` RPC with the true namespace plus signatures — and D-X9 deliberately keeps that out
 * of v1: it crosses into Rust and the wire for a fraction of the value. So: a name missing here is
 * NOT a name missing from `np`, and nothing should ever treat this list as the namespace.
 *
 * Grouped by completion `type` only so the popup can draw the right icon.
 */

/** Callables. */
const FUNCTIONS = [
	'abs',
	'angle',
	'arange',
	'arccos',
	'arcsin',
	'arctan',
	'arctan2',
	'argmax',
	'argmin',
	'argsort',
	'array',
	'asarray',
	'blackman',
	'ceil',
	'clip',
	'concatenate',
	'cos',
	'cosh',
	'cumsum',
	'degrees',
	'diff',
	'dot',
	'exp',
	'eye',
	'flip',
	'floor',
	'full',
	'hamming',
	'hanning',
	'hstack',
	'imag',
	'interp',
	'isfinite',
	'isnan',
	'linspace',
	'log',
	'log2',
	'log10',
	'max',
	'maximum',
	'mean',
	'median',
	'min',
	'minimum',
	'nanmean',
	'nanstd',
	'ones',
	'percentile',
	'power',
	'prod',
	'radians',
	'ravel',
	'real',
	'reshape',
	'roll',
	'round',
	'sign',
	'sin',
	'sinh',
	'sort',
	'sqrt',
	'square',
	'std',
	'sum',
	'tan',
	'tanh',
	'transpose',
	'unwrap',
	'var',
	'vstack',
	'where',
	'zeros'
];

/** Module-level constants. */
const CONSTANTS = ['e', 'inf', 'nan', 'newaxis', 'pi'];

/** Submodules that are imported with numpy itself. */
const NAMESPACES = ['fft', 'linalg', 'random'];

/** Every curated name with the completion `type` the popup draws it as. */
export const CURATED_NUMPY: { name: string; type: 'function' | 'constant' | 'namespace' }[] = [
	...FUNCTIONS.map((name) => ({ name, type: 'function' as const })),
	...CONSTANTS.map((name) => ({ name, type: 'constant' as const })),
	...NAMESPACES.map((name) => ({ name, type: 'namespace' as const }))
];
