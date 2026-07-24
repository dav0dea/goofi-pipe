/**
 * Pure layout resolvers for the flex primitives `Stack`/`Row` (spec §2.3, §3).
 *
 * `resolveSpace` maps a `gap` prop to a CSS length, and the align/justify maps turn the
 * short prop unions into their CSS flexbox values. Kept pure + unit-tested so the mapping
 * is one source of truth the components apply — never a literal spacing embedded per
 * component, always an F `--space-N` token.
 */

/** A spacing-scale key: `0` (no gap) or one of the F `--space-1`…`--space-8` steps. */
export type SpaceScale = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

/**
 * Resolve a `Stack`/`Row` `gap` prop to a CSS length.
 * - a numeric (or numeric-string) scale key 1–8 → the F `--space-N` token,
 * - `0` → the literal `0` (there is no `--space-0`),
 * - anything else (an already-resolved `var(--…)` token or an explicit length) → passed through.
 */
export function resolveSpace(gap: SpaceScale | string): string {
	if (gap === 0 || gap === '0') return '0';
	if (typeof gap === 'number') return `var(--space-${gap})`;
	if (/^[1-8]$/.test(gap)) return `var(--space-${gap})`;
	return gap;
}

/** `align` prop union → CSS `align-items` value. */
export type AlignSetting = 'stretch' | 'start' | 'center' | 'end' | 'baseline';
const ALIGN: Record<AlignSetting, string> = {
	stretch: 'stretch',
	start: 'flex-start',
	center: 'center',
	end: 'flex-end',
	baseline: 'baseline'
};
export function alignItems(align: AlignSetting): string {
	return ALIGN[align];
}

/** `justify` prop union → CSS `justify-content` value. */
export type JustifySetting = 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';
const JUSTIFY: Record<JustifySetting, string> = {
	start: 'flex-start',
	center: 'center',
	end: 'flex-end',
	between: 'space-between',
	around: 'space-around',
	evenly: 'space-evenly'
};
export function justifyContent(justify: JustifySetting): string {
	return JUSTIFY[justify];
}
