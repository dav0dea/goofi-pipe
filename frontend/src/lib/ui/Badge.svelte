<!--
  Badge — a small, static tone pill (spec §2.5): an uppercase, letter-spaced micro label that
  colour-codes status/kind at a glance. Non-interactive (a plain <span>); its pressable sibling is
  <Chip>, which shares this tone scale. Each tone paints a tinted fill + a legible same-hue text from
  the F semantic tokens — neutral leans on the surface/dim pair, and the four coloured tones reuse the
  ready-made --accent-fill/--danger-fill/--success-fill/--warning-fill (one source per tint). `class`
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
	<!-- The ink wrapper is the glyph run on its own — the seam the gallery's ink pin selects to
	     measure where the visible ink sits against the pill box. It once carried a translateY
	     correction; the real vendored faces centre by their own metrics, so it carries none. -->
	<span class="ui-badge-ink">{@render children?.()}</span>
</span>

<style>
	.ui-badge {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
		padding: var(--space-1) var(--space-3);
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		font-family: var(--font-sans);
		font-size: var(--fs-micro);
		font-weight: 600;
		line-height: 1;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		white-space: nowrap;
	}
	.ui-badge-ink {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
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
		background: var(--success-fill);
		border-color: color-mix(in srgb, var(--success) 40%, transparent);
		color: var(--success);
	}
	.ui-badge.t-warning {
		background: var(--warning-fill);
		border-color: color-mix(in srgb, var(--warning) 40%, transparent);
		color: var(--warning);
	}
	.ui-badge.t-danger {
		background: var(--danger-fill);
		border-color: color-mix(in srgb, var(--danger) 40%, transparent);
		color: var(--danger);
	}
</style>
