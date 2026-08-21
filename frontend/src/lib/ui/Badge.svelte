<!-- Badge — a small, static tone pill; its pressable sibling is <Chip>, on the same tone scale. -->
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
	<!-- The ink wrapper is the glyph run alone — the seam the gallery's ink pin measures. -->
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
