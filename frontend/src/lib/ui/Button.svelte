<!--
  Button — the foundation interactive primitive (spec §2.1).

  Fully self-styled: its own `.ui-btn` scoped class specifies background/border/padding/
  radius/font from F tokens, so it renders correctly independent of `app.css`'s base
  `button` rule, which keeps only a `font: inherit` + `cursor` RESET (the skin went at M-Task 7;
  the reset is permanent — see app.css, and the C19 box-geometry test in ui-gallery.spec.ts). Variant
  + size come from the pure `variantClass` map; `class` is merged (not replaced) and every
  other attribute — `disabled`, `onclick`, `data-testid`, `title`, aria-* — forwards through.

  Touch: as a real <button>, it inherits the app-wide coarse-pointer `min-height: var(--hit)`
  floor from app.css. Keyboard focus rings via the app-wide `:focus-visible` (never suppressed).
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
		font-family: var(--font-sans);
		/* The app body ratio (app.css `body`), stated rather than inherited (C19). The box height IS
		   this plus the padding and border, and app.css's base `button` rule keeps only a `font:
		   inherit` reset — so leaving it implicit makes every Button's height a property of whatever
		   it happens to be nested in, and `normal` (the UA value under any stricter reset) shortens
		   the lot by 1-2px. `s-md`/`s-sm` scale it by setting only `font-size`. */
		line-height: var(--lh-text);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-2);
		color: var(--text);
		cursor: pointer;
		white-space: nowrap;
		transition:
			background var(--dur-fast) var(--ease),
			border-color var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	/* The other axis. app.css floors the HEIGHT of every button under a coarse pointer, but a short
	   label ("Kill") is 40px wide there — and the blanket rule cannot carry the width, because three
	   frozen node-canvas controls defeat the floor on purpose. A Button is never one of them. */
	@media (hover: none) and (pointer: coarse) {
		.ui-btn {
			min-width: var(--hit);
		}
	}
	.ui-btn:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}

	/* Size — padding + type scale from the F step ladder. */
	.ui-btn.s-md {
		padding: var(--space-3) var(--space-6);
		font-size: var(--fs-small);
	}
	.ui-btn.s-sm {
		padding: var(--space-2) var(--space-4);
		font-size: var(--fs-micro);
	}

	/* Variants — colour only; the default is the resting surface. Hover is an
	   enhancement, never the sole affordance (the control is always visible + clickable). */
	.ui-btn.v-default:hover:not(:disabled) {
		background: var(--surface-3);
		border-color: var(--border-strong);
	}
	.ui-btn.v-primary {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--on-accent);
		font-weight: 600;
	}
	.ui-btn.v-primary:hover:not(:disabled) {
		background: var(--accent-strong);
		border-color: var(--accent-strong);
	}
	.ui-btn.v-ghost {
		background: transparent;
		border-color: transparent;
		/* A ghost is ink on someone else's surface, so its ink is the one thing a host may need to
		   restate — a status glyph in a chrome strip carries its meaning in its colour, not in a
		   fill. Per-instance hook, unset it resolves to the same `--text` every other variant uses. */
		color: var(--btn-ink, var(--text));
	}
	/* A ghost has no surface of its own, so its hover LIFTS its host rather than naming a rung —
	   `--surface-2` was invisible on every chrome strip these actually sit on (see app.css). */
	.ui-btn.v-ghost:hover:not(:disabled) {
		background: var(--hover-fill);
	}
	.ui-btn.v-danger {
		background: var(--danger);
		border-color: var(--danger);
		color: var(--on-danger);
		font-weight: 600;
	}
	.ui-btn.v-danger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--danger) 85%, var(--bg));
		border-color: color-mix(in srgb, var(--danger) 85%, var(--bg));
	}
</style>
