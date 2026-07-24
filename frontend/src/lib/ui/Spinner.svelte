<!--
  Spinner — a CSS ring spinner (spec §2.5): a border ring with one accented quadrant rotating over a
  neutral track — the same shape as the node boot-spinner. `size` picks the diameter (rem, so it
  tracks the responsive type clamp); the ring colour is --accent over a --border track, both
  per-instance overridable (--spinner-size / --spinner-track / --spinner-color).

  The global prefers-reduced-motion rule already all-but-freezes the loop (duration 0.01ms, one
  iteration); the guard below stops it outright so the ring's reduced-motion behaviour is self-evident
  here and does not silently depend on that distant global. Presentational: aria-hidden by default;
  pass an `aria-label` to voice a loading state. `class` merged, `data-testid` (+ any attribute) forwarded.
-->
<script module lang="ts">
	export type SpinnerSize = 'sm' | 'md';
</script>

<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		size = 'md',
		'aria-label': ariaLabel,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLSpanElement> & {
		size?: SpinnerSize;
	} = $props();
</script>

<span
	{...rest}
	role={ariaLabel ? 'status' : undefined}
	aria-label={ariaLabel}
	aria-hidden={ariaLabel ? undefined : true}
	class={`ui-spinner s-${size} ${klass}`.trim()}
></span>

<style>
	.ui-spinner {
		display: inline-block;
		flex-shrink: 0;
		box-sizing: border-box;
		width: var(--spinner-size);
		height: var(--spinner-size);
		border: 2px solid var(--spinner-track, var(--border));
		border-top-color: var(--spinner-color, var(--accent));
		border-radius: 50%;
		animation: ui-spin 0.8s linear infinite;
	}
	.ui-spinner.s-sm {
		--spinner-size: 0.85rem;
	}
	.ui-spinner.s-md {
		--spinner-size: 1.25rem;
	}
	@keyframes ui-spin {
		to {
			transform: rotate(360deg);
		}
	}
	/* Stop the loop outright — clearer here than leaning on the global reduced-motion longhands. */
	@media (prefers-reduced-motion: reduce) {
		.ui-spinner {
			animation: none;
		}
	}
</style>
