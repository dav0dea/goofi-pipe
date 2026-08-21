<!-- Chip — the pressable sibling of <Badge>: the same tone pill as a real <button>. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import type { BadgeTone } from './Badge.svelte';
	import type { ButtonDensity } from 'panelty';

	let {
		tone = 'neutral',
		density = 'comfortable',
		type = 'button',
		class: klass = '',
		children,
		...rest
	}: HTMLButtonAttributes & {
		tone?: BadgeTone;
		/** Box density; `chrome` is the dense box a toolbar strip wears. */
		density?: ButtonDensity;
		children?: Snippet;
	} = $props();
</script>

<button
	{...rest}
	{type}
	class={`ui-chip t-${tone} ${density === 'chrome' ? 'd-chrome ' : ''}${klass}`.trim()}
>
	<!-- Same ink wrapper as Badge: the glyph run alone, for the gallery's ink pin to measure. -->
	<span class="ui-chip-ink">{@render children?.()}</span>
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
		font-family: var(--font-sans);
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
	/* Unset, the hook falls back to --hit, so `density="chrome"` alone is a no-op, not a collapse. */
	.ui-chip.d-chrome {
		min-height: var(--chip-size, var(--hit));
	}
	/* The width floor lives here, not in app.css: a blanket one there widens the frozen node-canvas
	   exceptions. */
	@media (hover: none) and (pointer: coarse) {
		.ui-chip {
			min-width: var(--hit);
			justify-content: center;
		}
		.ui-chip.d-chrome {
			min-height: var(--hit);
		}
	}
	.ui-chip-ink {
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
	}
	.ui-chip:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
	.ui-chip:focus-visible {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: 2px;
	}

	/* `neutral` is a ghost, unlike Badge's filled one: it is the RESTING state and must not outshout
	   the label it sits beside. */
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
