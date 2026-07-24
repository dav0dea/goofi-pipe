<!--
  Bar — a horizontal toolbar / header strip (spec §2.3), the "pusher pattern": a `start` snippet
  group hugs the left, a flex spacer eats the slack, and an `end` snippet group is pushed to the
  right. Either group is optional (the spacer alone still pushes a lone `end` right / keeps a lone
  `start` left). Merged `class`, forwarded `data-testid` (and any other attribute) via `...rest`.

  Padding, height, gap and chrome are all F tokens, each exposed as a `var(--bar-*, <token>)` hook
  so a consumer can retune one instance from a wrapper — the spec §1 per-instance-theming mechanism.
  Defaults give the panel-header look (surface-1 strip, hairline bottom border, `--panel-header-h`
  min-height that grows to the touch floor under a coarse pointer).
-->
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
		min-width: 0;
		min-height: var(--bar-height, var(--panel-header-h));
		padding: var(--bar-pad-y, var(--space-2)) var(--bar-pad-x, var(--space-4));
		gap: var(--bar-gap, var(--space-4));
		background: var(--bar-bg, var(--surface-1));
		border-bottom: var(--bar-border, 1px solid var(--border));
	}
	/* Each group lays its own items in a row and can shrink (so a long label ellipsis's, not overflows). */
	.ui-bar-group {
		display: flex;
		align-items: center;
		gap: var(--bar-gap, var(--space-4));
		min-width: 0;
	}
	/* The pusher: eats the slack between start and end, forcing end to the right edge. */
	.ui-bar-spacer {
		flex: 1 1 auto;
	}
</style>
