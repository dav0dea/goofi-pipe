/**
 * Pure variant→class mapper for the button primitives (spec §3).
 *
 * Returns the space-joined scoped class names a `<Button>`/`<IconButton>` applies to
 * its root element; each component's own scoped `<style>` defines `.v-*`/`.s-*` from F
 * tokens. Kept pure + unit-tested so the variant surface is one closed union with a
 * single source of truth, not re-derived (and drifting) per component.
 */
export type ButtonVariant = 'default' | 'primary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';
/**
 * Box density — orthogonal to `size` (which sets the *type scale* of the label/glyph).
 *
 * `comfortable` is the default box every call site gets: floored to the F `--tatami-hit` target.
 * `chrome` is the dense box a window-chrome strip needs — a tab bar or a panel header, whose
 * own height is smaller than `--tatami-hit` — sized per instance via the `--tatami-icon-btn-size` hook on a
 * fine pointer and floored back to `--tatami-hit` under a coarse one. Emitting no class for the
 * default keeps a comfortable control's class list exactly as it was.
 */
export type ButtonDensity = 'comfortable' | 'chrome';

export function variantClass(
	variant: ButtonVariant,
	size: ButtonSize,
	density: ButtonDensity = 'comfortable'
): string {
	const base = `v-${variant} s-${size}`;
	return density === 'comfortable' ? base : `${base} d-${density}`;
}
