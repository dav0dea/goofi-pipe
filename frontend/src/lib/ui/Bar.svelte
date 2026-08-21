<!-- Bar — a toolbar strip: a `start` group, a spacer that eats the slack, an `end` group pushed
     right. Every measure is a `var(--bar-*, <token>)` hook, so one instance can be retuned. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		start,
		end,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		/** Left-hugging group. */
		start?: Snippet;
		/** Right-pushed group. */
		end?: Snippet;
	} = $props();
</script>

<div {...rest} class={`ui-bar ${klass}`.trim()}>
	{#if start}
		<div class="ui-bar-group">{@render start()}</div>
	{/if}
	<div class="ui-bar-spacer"></div>
	{#if end}
		<div class="ui-bar-group">{@render end()}</div>
	{/if}
</div>

<style>
	.ui-bar {
		display: flex;
		align-items: center;
		flex-wrap: var(--bar-wrap, nowrap);
		min-width: 0;
		min-height: var(--bar-height, var(--panel-header-h));
		/* No vertical padding: the strip is exactly the panel-header height, and a bar that needs
		   some asks for it back through `--bar-pad-y`. */
		padding: var(--bar-pad-y, 0) var(--bar-pad-x, var(--space-4));
		/* Inherited, not reached into: a control is dense by being IN a bar, not by saying so. */
		--panelty-icon-btn-size: var(--chrome-control-h);
		--chip-size: var(--chrome-control-h);
		gap: var(--bar-gap, var(--space-4));
		background: var(--bar-bg, var(--surface-2));
		border-bottom: var(--bar-border, none);
	}
	.ui-bar-group {
		display: flex;
		align-items: center;
		gap: var(--bar-gap, var(--space-4));
		min-width: 0;
	}
	/* Keeps the `end` group right-aligned once it has wrapped past the spacer. */
	.ui-bar-group:last-child {
		margin-left: auto;
		/* A bar with unshrinkable end actions sets `--bar-end-min: max-content`, so the start
		   label yields instead of the actions overflowing. */
		min-width: var(--bar-end-min, 0);
	}
	.ui-bar-spacer {
		flex: 1 1 auto;
	}
</style>
