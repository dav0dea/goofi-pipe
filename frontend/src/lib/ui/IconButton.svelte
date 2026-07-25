<!--
  IconButton — a square glyph action (spec §2.1). Same variant/size palette as Button,
  fully self-styled from F tokens. A `label` is required and becomes the `aria-label`
  (and default `title`) since the visible content is a glyph, not text.

  Touch hit-targeting (deferred from F): the box is floored to the F `--hit` target
  (28px fine / 44px coarse) so its rendered bounding box is a real tap target while the
  glyph stays visually small. Additionally, under a coarse pointer an `::after` overlay
  guarantees a >= --hit clickable area even if a consumer shrinks the visual box below the
  floor; under a fine pointer that overlay does not exist (a genuine no-op).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import { variantClass, type ButtonVariant, type ButtonSize } from './variantClass';

	let {
		variant = 'default',
		size = 'md',
		type = 'button',
		label,
		class: klass = '',
		children,
		...rest
	}: HTMLButtonAttributes & {
		variant?: ButtonVariant;
		size?: ButtonSize;
		/** Accessible name — required; the visible content is a glyph. */
		label: string;
		children?: Snippet;
	} = $props();
</script>

<button
	{...rest}
	{type}
	class={`ui-icon-btn ${variantClass(variant, size)} ${klass}`.trim()}
	aria-label={label}
	title={rest.title ?? label}
>
	<span class="glyph">{@render children?.()}</span>
</button>

<style>
	.ui-icon-btn {
		/* A <button> inherits no font of its own — without this the glyph would fall back to the
		   UA face once M-Task 7 strips app.css's base `button` rule. `line-height` below still
		   wins over the shorthand's inherited one (later declaration). */
		font: inherit;
		position: relative; /* anchor for the coarse-pointer hit-rect ::after */
		display: inline-grid;
		place-items: center;
		/* Square tap target floored to the F hit geometry, so the rendered box IS a real
		   touch target (28 fine / 44 coarse) while the glyph inside stays small. */
		min-width: var(--hit);
		min-height: var(--hit);
		padding: 0;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-2);
		color: var(--text);
		cursor: pointer;
		line-height: 1;
		transition:
			background var(--dur-fast) var(--ease),
			border-color var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-icon-btn:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}

	/* The glyph stays visually small regardless of the tap-target size. */
	.glyph {
		display: inline-grid;
		place-items: center;
		pointer-events: none;
	}
	.ui-icon-btn.s-md .glyph {
		font-size: var(--fs-body);
	}
	.ui-icon-btn.s-sm .glyph {
		font-size: var(--fs-small);
	}

	/* Variants — colour only. Hover is an enhancement, never the sole affordance. */
	.ui-icon-btn.v-default:hover:not(:disabled) {
		background: var(--surface-3);
		border-color: var(--border-strong);
	}
	.ui-icon-btn.v-primary {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--bg);
	}
	.ui-icon-btn.v-primary:hover:not(:disabled) {
		background: var(--accent-strong);
		border-color: var(--accent-strong);
	}
	.ui-icon-btn.v-ghost {
		background: transparent;
		border-color: transparent;
	}
	.ui-icon-btn.v-ghost:hover:not(:disabled) {
		background: var(--surface-2);
	}
	.ui-icon-btn.v-danger {
		background: var(--danger);
		border-color: var(--danger);
		color: var(--bg);
	}
	.ui-icon-btn.v-danger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--danger) 85%, var(--bg));
		border-color: color-mix(in srgb, var(--danger) 85%, var(--bg));
	}

	/* Coarse-pointer hit-rect guarantee: extend the clickable area outward to at least
	   --hit. A no-op once the visual box already meets --hit (inset resolves to 0); only
	   grows the target if a consumer shrinks the box below the floor. Absent under a fine
	   pointer, so there it is a genuine no-op. */
	@media (pointer: coarse) {
		.ui-icon-btn::after {
			content: '';
			position: absolute;
			inset: calc((var(--hit) - 100%) / -2);
		}
	}
</style>
