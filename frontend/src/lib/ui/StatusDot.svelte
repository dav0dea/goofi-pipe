<!--
  StatusDot — one status dot (spec §2.5): a plain filled circle whose `tone` colour-codes health
  (ok/error/warn → the F --success/--danger/--warning semantics) at a small size scale. Deliberately
  NO glow — F removed the health-dot's box-shadow halo, so this dot carries NONE: a flat disc, never a
  beacon. Diameter is per-instance overridable (--status-dot-size) and sized in rem so it tracks the
  responsive type clamp (bigger on a coarse-pointer phone). Presentational: aria-hidden by default;
  pass an `aria-label` to voice it. `class` merged, `data-testid` (+ any attribute) forwarded.
-->
<script module lang="ts">
	export type StatusTone = 'ok' | 'error' | 'warn';
	export type StatusDotSize = 'sm' | 'md';
</script>

<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		tone,
		size = 'md',
		pulse = false,
		'aria-label': ariaLabel,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLSpanElement> & {
		tone: StatusTone;
		size?: StatusDotSize;
		/** Blink instead of resting — for a state that is not merely bad but GONE. */
		pulse?: boolean;
	} = $props();
</script>

<span
	{...rest}
	aria-label={ariaLabel}
	aria-hidden={ariaLabel ? undefined : true}
	class={`ui-status-dot t-${tone} s-${size}${pulse ? ' pulse' : ''} ${klass}`.trim()}
></span>

<style>
	.ui-status-dot {
		display: inline-block;
		flex-shrink: 0;
		width: var(--status-dot-size);
		height: var(--status-dot-size);
		border-radius: 50%;
		/* NO box-shadow: the health-dot glow was removed in F and must not return. */
	}
	.ui-status-dot.s-sm {
		--status-dot-size: 0.5rem;
	}
	.ui-status-dot.s-md {
		--status-dot-size: 0.625rem;
	}
	.ui-status-dot.t-ok {
		background: var(--success);
	}
	.ui-status-dot.t-error {
		background: var(--danger);
	}
	.ui-status-dot.t-warn {
		background: var(--warning);
	}
	/* Opacity only — the dot carries no glow (above) and must not grow one to blink. */
	.ui-status-dot.pulse {
		animation: status-blink 1.1s steps(1, end) infinite;
	}
	@keyframes status-blink {
		0%,
		50% {
			opacity: 1;
		}
		50.01%,
		100% {
			opacity: 0.2;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.ui-status-dot.pulse {
			animation: none;
			opacity: 0.6;
		}
	}
</style>
