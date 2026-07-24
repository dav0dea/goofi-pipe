<!--
  Button — the foundation interactive primitive (spec §2.1).

  Fully self-styled: its own `.ui-btn` scoped class specifies background/border/padding/
  radius/font from F tokens, so it renders correctly whether or not `app.css`'s base
  `button` rule exists (M strips that rule last, after migrating every call site). Variant
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
		font-family: var(--font-mono);
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
		color: var(--bg);
		font-weight: 600;
	}
	.ui-btn.v-primary:hover:not(:disabled) {
		background: var(--accent-strong);
		border-color: var(--accent-strong);
	}
	.ui-btn.v-ghost {
		background: transparent;
		border-color: transparent;
	}
	.ui-btn.v-ghost:hover:not(:disabled) {
		background: var(--surface-2);
	}
	.ui-btn.v-danger {
		background: var(--danger);
		border-color: var(--danger);
		color: var(--bg);
		font-weight: 600;
	}
	.ui-btn.v-danger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--danger) 85%, var(--bg));
		border-color: color-mix(in srgb, var(--danger) 85%, var(--bg));
	}
</style>
