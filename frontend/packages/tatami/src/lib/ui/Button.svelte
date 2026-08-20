<!--
  Button — the foundation interactive primitive (spec §2.1).

  Fully self-styled: its own `.ui-btn` scoped class specifies background/border/padding/
  radius/font from F tokens, so it renders correctly independent of `app.css`'s base
  `button` rule, which keeps only a `font: inherit` + `cursor` RESET (the skin went at M-Task 7;
  the reset is permanent — see app.css, and the C19 box-geometry test in ui-gallery.spec.ts). Variant
  + size come from the pure `variantClass` map; `class` is merged (not replaced) and every
  other attribute — `disabled`, `onclick`, `data-testid`, `title`, aria-* — forwards through.

  Touch: under a coarse pointer the box is floored to `--tatami-hit` on BOTH axes, stated below
  rather than inherited from a host app's blanket `button {}` reset — a package that only looks
  right inside one app is not one. Keyboard focus rings via `:focus-visible` (never suppressed).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import { variantClass, type ButtonVariant, type ButtonSize } from './variantClass';

	let {
		variant = 'default',
		size = 'md',
		type = 'button',
		class: klass = '',
		children,
		...rest
	}: HTMLButtonAttributes & {
		variant?: ButtonVariant;
		size?: ButtonSize;
		children?: Snippet;
	} = $props();
</script>

<button {...rest} {type} class={`ui-btn ${variantClass(variant, size)} ${klass}`.trim()}>
	{@render children?.()}
</button>

<style>
	.ui-btn {
		font-family: var(--tatami-font-sans, var(--tatami-font-sans-default));
		/* The app body ratio (app.css `body`), stated rather than inherited (C19). The box height IS
		   this plus the padding and border, and app.css's base `button` rule keeps only a `font:
		   inherit` reset — so leaving it implicit makes every Button's height a property of whatever
		   it happens to be nested in, and `normal` (the UA value under any stricter reset) shortens
		   the lot by 1-2px. `s-md`/`s-sm` scale it by setting only `font-size`. */
		line-height: var(--tatami-lh-text, var(--tatami-lh-text-default));
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--tatami-space-3, var(--tatami-space-3-default));
		border: 1px solid var(--tatami-border, var(--tatami-border-default));
		border-radius: var(--tatami-radius-sm, var(--tatami-radius-sm-default));
		background: var(--tatami-surface-2, var(--tatami-surface-2-default));
		color: var(--tatami-text, var(--tatami-text-default));
		cursor: pointer;
		white-space: nowrap;
		transition:
			background var(--tatami-motion, var(--tatami-motion-default)),
			border-color var(--tatami-motion, var(--tatami-motion-default)),
			color var(--tatami-motion, var(--tatami-motion-default));
	}
	/* BOTH axes: a short label ("Kill") is 40px wide under a coarse pointer, so a height floor alone
	   leaves it under the target. An app's own blanket `button {}` rule may floor the height too —
	   goofi's does — and lands on the same token, so this restates rather than fights it. */
	@media (hover: none) and (pointer: coarse) {
		.ui-btn {
			min-width: var(--tatami-hit, var(--tatami-hit-default));
			min-height: var(--tatami-hit, var(--tatami-hit-default));
		}
	}
	.ui-btn:disabled {
		opacity: var(--tatami-disabled-opacity, var(--tatami-disabled-opacity-default));
		cursor: not-allowed;
	}

	/* Size — padding + type scale from the F step ladder. */
	.ui-btn.s-md {
		padding:
			var(--tatami-space-3, var(--tatami-space-3-default))
			var(--tatami-space-6, var(--tatami-space-6-default));
		font-size: var(--tatami-fs-small, var(--tatami-fs-small-default));
	}
	.ui-btn.s-sm {
		padding:
			var(--tatami-space-2, var(--tatami-space-2-default))
			var(--tatami-space-4, var(--tatami-space-4-default));
		font-size: var(--tatami-fs-micro, var(--tatami-fs-micro-default));
	}

	/* Variants — colour only; the default is the resting surface. Hover is an
	   enhancement, never the sole affordance (the control is always visible + clickable). */
	.ui-btn.v-default:hover:not(:disabled) {
		background: var(--tatami-surface-3, var(--tatami-surface-3-default));
		border-color: var(--tatami-border-strong, var(--tatami-border-strong-default));
	}
	.ui-btn.v-primary {
		background: var(--tatami-accent, var(--tatami-accent-default));
		border-color: var(--tatami-accent, var(--tatami-accent-default));
		color: var(--tatami-on-accent, var(--tatami-on-accent-default));
		font-weight: 600;
	}
	.ui-btn.v-primary:hover:not(:disabled) {
		background: var(--tatami-accent-strong, var(--tatami-accent-strong-default));
		border-color: var(--tatami-accent-strong, var(--tatami-accent-strong-default));
	}
	.ui-btn.v-ghost {
		background: transparent;
		border-color: transparent;
		/* A ghost is ink on someone else's surface, so its ink is the one thing a host may need to
		   restate — a status glyph in a chrome strip carries its meaning in its colour, not in a
		   fill. Per-instance hook, unset it resolves to the same `--tatami-text` every other variant uses. */
		color: var(--tatami-btn-ink, var(--tatami-text, var(--tatami-text-default)));
	}
	/* A ghost has no surface of its own, so its hover LIFTS its host rather than naming a rung —
	   `--tatami-surface-2` was invisible on every chrome strip these actually sit on (see app.css). */
	.ui-btn.v-ghost:hover:not(:disabled) {
		background: var(--tatami-hover-fill, var(--tatami-hover-fill-default));
	}
	.ui-btn.v-danger {
		background: var(--tatami-danger, var(--tatami-danger-default));
		border-color: var(--tatami-danger, var(--tatami-danger-default));
		color: var(--tatami-on-danger, var(--tatami-on-danger-default));
		font-weight: 600;
	}
	.ui-btn.v-danger:hover:not(:disabled) {
		background: color-mix(
			in srgb,
			var(--tatami-danger, var(--tatami-danger-default)) 85%,
			var(--tatami-bg, var(--tatami-bg-default))
		);
		border-color: color-mix(
			in srgb,
			var(--tatami-danger, var(--tatami-danger-default)) 85%,
			var(--tatami-bg, var(--tatami-bg-default))
		);
	}
</style>
