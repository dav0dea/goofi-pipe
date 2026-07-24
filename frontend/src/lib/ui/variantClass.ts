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

export function variantClass(variant: ButtonVariant, size: ButtonSize): string {
	return `v-${variant} s-${size}`;
}
