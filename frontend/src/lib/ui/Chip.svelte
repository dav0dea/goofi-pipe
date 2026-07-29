<!--
  Chip — the pressable sibling of <Badge> (spec §2.5): the same uppercase tone pill, but a real
  <button> that fires `onclick`. Reuses Badge's tone scale (imported so the two never drift) and its
  visual language, adding an interactive layer — a hover that warms the fill, a :focus-visible ring,
  and the coarse-pointer --hit floor so it is a legal touch target. Fully self-styled from F tokens
  (it does NOT lean on the app.css base `button` rule; its own class specifies everything). `onclick`
  (and every other button attribute) forwards via `...rest`; `class` merged; `data-testid` forwarded.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import type { BadgeTone } from './Badge.svelte';

	let {
		tone = 'neutral',
		type = 'button',
		class: klass = '',
		children,
		...rest
	}: HTMLButtonAttributes & {
		tone?: BadgeTone;
		children?: Snippet;
	} = $props();
</script>

<button {...rest} {type} class={`ui-chip t-${tone} ${klass}`.trim()}>
	{@render children?.()}
</button>

<style>
	.ui-chip {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		min-height: var(--hit);
		padding: var(--space-1) var(--space-4);
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		font-weight: 600;
		line-height: 1;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		white-space: nowrap;
		cursor: pointer;
		transition:
			background var(--dur-fast) var(--ease),
			border-color var(--dur-fast) var(--ease),
			color var(--dur-fast) var(--ease);
	}
	.ui-chip:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
	.ui-chip:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: 2px;
	}

	/* Tones mirror Badge's fills; hover intensifies the tint (never the sole affordance — the pill is
	   always visible + clickable, and keyboard focus rings it).

	   `neutral` is the exception, and deliberately so: it is the RESTING state, and most Chips in
	   this app rest — the inspector's `fx` toggle is on every parameter row in every state, and the
	   console's `out`/`err` filters rest off. As a fill plus a hairline it made the loudest element
	   on a param label row the toggle rather than the parameter NAME, which the inspector's own
	   spec calls the primary scan target. So it is a ghost, the same shape `Button`/`IconButton`
	   already ship, hovering onto the shared `--hover-fill` that exists for exactly this — a
	   control with no surface of its own, which must lift whatever rung it happens to sit on. Badge
	   keeps its filled neutral: it is a static label, not a control at rest. */
	.ui-chip.t-neutral {
		background: transparent;
		border-color: transparent;
		color: var(--text-muted);
	}
	.ui-chip.t-neutral:hover:not(:disabled) {
		background: var(--hover-fill);
		color: var(--text);
	}
	.ui-chip.t-accent {
		background: var(--accent-fill);
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
		color: var(--accent);
	}
	.ui-chip.t-accent:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 28%, transparent);
		border-color: var(--accent);
	}
	.ui-chip.t-success {
		background: var(--success-fill);
		border-color: color-mix(in srgb, var(--success) 40%, transparent);
		color: var(--success);
	}
	.ui-chip.t-success:hover:not(:disabled) {
		background: color-mix(in srgb, var(--success) 28%, transparent);
		border-color: var(--success);
	}
	.ui-chip.t-warning {
		background: var(--warning-fill);
		border-color: color-mix(in srgb, var(--warning) 40%, transparent);
		color: var(--warning);
	}
	.ui-chip.t-warning:hover:not(:disabled) {
		background: color-mix(in srgb, var(--warning) 28%, transparent);
		border-color: var(--warning);
	}
	.ui-chip.t-danger {
		background: var(--danger-fill);
		border-color: color-mix(in srgb, var(--danger) 40%, transparent);
		color: var(--danger);
	}
	.ui-chip.t-danger:hover:not(:disabled) {
		background: color-mix(in srgb, var(--danger) 28%, transparent);
		border-color: var(--danger);
	}
</style>
