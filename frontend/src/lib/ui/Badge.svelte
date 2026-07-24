<!--
  Badge — a small, static tone pill (spec §2.5): an uppercase, letter-spaced micro label that
  colour-codes status/kind at a glance. Non-interactive (a plain <span>); its pressable sibling is
  <Chip>, which shares this tone scale. Each tone paints a tinted fill + a legible same-hue text from
  the F semantic tokens — neutral leans on the surface/dim pair, accent/danger reuse the ready-made
  --accent-fill/--danger-fill, and success/warning mix their own tint in the same idiom. `class`
  merged (not replaced), `data-testid` (and any other attribute) forwarded via `...rest`.
-->
<script module lang="ts">
	export type BadgeTone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger';
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		tone = 'neutral',
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLSpanElement> & {
		tone?: BadgeTone;
		children?: Snippet;
	} = $props();
</script>

<span {...rest} class={`ui-badge t-${tone} ${klass}`.trim()}>
	{@render children?.()}
</span>

<style>
	.ui-badge {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-1) var(--space-3);
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		font-weight: 600;
		line-height: 1;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		white-space: nowrap;
	}
	/* Neutral — the resting surface pill: dim text on a raised surface, hairline border. */
	.ui-badge.t-neutral {
		background: var(--surface-3);
		border-color: var(--border);
		color: var(--text-dim);
	}
	.ui-badge.t-accent {
		background: var(--accent-fill);
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
		color: var(--accent);
	}
	.ui-badge.t-success {
		background: color-mix(in srgb, var(--success) 16%, transparent);
		border-color: color-mix(in srgb, var(--success) 40%, transparent);
		color: var(--success);
	}
	.ui-badge.t-warning {
		background: color-mix(in srgb, var(--warning) 16%, transparent);
		border-color: color-mix(in srgb, var(--warning) 40%, transparent);
		color: var(--warning);
	}
	.ui-badge.t-danger {
		background: var(--danger-fill);
		border-color: color-mix(in srgb, var(--danger) 40%, transparent);
		color: var(--danger);
	}
</style>
