<!-- StatusDot — a flat filled circle whose `tone` colour-codes health; aria-hidden unless an
     `aria-label` is passed. -->
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
		/** Blink instead of resting: a state that is not merely bad but GONE. */
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
		/* No box-shadow: this dot is a flat disc, never a beacon. */
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
